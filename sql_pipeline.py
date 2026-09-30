"""
The actual clarification-engineering pipeline.

check_ambiguity()   -> schema-aware LLM ambiguity detection
                        with dynamically generated clarification options

generate_sql()      -> schema-aware SQL generation, clarified intent baked
                        into the prompt when applicable

validate_sql()      -> sqlglot-based safety check (SELECT-only, single
                        statement, LIMIT enforced)
"""
from __future__ import annotations
from typing import Optional

import sqlglot
from sqlglot import exp

from models import (
    AmbiguityCheck,
    AmbiguityOption,
    ClarificationOption,
    ClarificationRequest,
    DatabaseSchema,
    SQLGenerationResult,
    ValidationResult,
)
import llm_router

DEFAULT_ROW_LIMIT = 200

def _schema_columns(schema: DatabaseSchema) -> set[str]:
    """
    Return fully qualified and unqualified column names available
    in the schema.
    """
    columns = set()

    for table in schema.tables:
        for column in table.columns:
            columns.add(column.name.lower())
            columns.add(f"{table.name}.{column.name}".lower())

    return columns
def _option_uses_known_columns(
    option: AmbiguityOption,
    schema: DatabaseSchema,
) -> bool:
    """
    Deterministically verify that every column referenced by an
    ambiguity option actually exists in the live database schema.
    """

    if not option.column_refs:
        return False

    known = _schema_columns(schema)

    for ref in option.column_refs:
        ref = ref.strip().lower()

        if ref not in known:
            return False

    return True

def _option_is_schema_grounded(
    option: AmbiguityOption,
    schema: DatabaseSchema,
) -> bool:
    """
    Validate that a clarification option is grounded in the
    actual database schema.
    """

    # Must provide explicit schema references
    if not option.column_refs:
        return False

    # All referenced columns must exist
    if not _option_uses_known_columns(option, schema):
        return False

    # Must explain why the option is possible
    if not option.schema_basis.strip():
        return False

    # Must provide a SQL interpretation
    if not option.sql_hint.strip():
        return False

    return True
# ---------------------------------------------------------------------------
# 1. Ambiguity detection
# ---------------------------------------------------------------------------

def check_ambiguity(
    query: str,
    schema: DatabaseSchema,
    provider: str,
    api_key: Optional[str],
) -> Optional[ClarificationRequest]:

    if not api_key:
        return None

    system = """
You are a schema-aware ambiguity detector for a Text-to-SQL system.

Your job is NOT to generate SQL.

Use ONLY the database schema provided to you.

Rules:

1. Identify genuinely ambiguous business language.
2. Only generate interpretations that can actually be computed
   from the provided schema.
3. Never invent tables or columns.
4. Every option must contain:
   - label
   - sql_hint
   - column_refs
   - schema_basis
5. column_refs must contain exact table.column names from the schema.
6. schema_basis must explain why the option is computable.
7. Generate 2-4 options when ambiguity exists.
8. If the question is sufficiently precise, return is_ambiguous=false.
9. Do not write complete SQL queries.

Example:

Question:
"Who is the best customer?"

If the schema contains:

customers(id, name)
orders(id, customer_id, total_amount)
returns(id, customer_id)

valid interpretations include:

- Highest total spending
- Most orders placed
- Fewest returns

But do NOT invent metrics whose required columns do not exist.
"""

    user = f"""
DATABASE SCHEMA:

{schema.as_prompt_block()}

USER QUESTION:

{query}
"""

    try:
        result = llm_router.call_structured(
            provider=provider,
            api_key=api_key,
            system_prompt=system,
            user_prompt=user,
            schema=AmbiguityCheck,
        )
    except Exception:
        return None

    if not result.is_ambiguous:
        return None

    if not result.options:
        return None

    valid_options = [
        option
        for option in result.options
        if _option_is_schema_grounded(option, schema)
    ]

    if not valid_options:
        return None

    options = [
        ClarificationOption(
            label=option.label,
            sql_hint=option.sql_hint,
            column_ref=", ".join(option.column_refs),
        )
        for option in valid_options
    ]

    return ClarificationRequest(
        original_query=query,
        question=result.question
        or "Could you clarify what you mean?",
        options=options,
        source="schema_llm",
    )



# ---------------------------------------------------------------------------
# 2. SQL generation
# ---------------------------------------------------------------------------

def generate_sql(
    query: str,
    schema: DatabaseSchema,
    provider: str,
    api_key: str,
    resolved_hint: Optional[str] = None,
    dialect: str = "duckdb",
    previous_error: Optional[str] = None,
    previous_sql: Optional[str] = None,
) -> SQLGenerationResult:
    system = (
        f"You are an expert SQL analyst. Generate a single read-only SQL query "
        f"({dialect} dialect) that answers the user's question, using ONLY the "
        f"tables and columns given in the schema. Always include a LIMIT clause "
        f"(default {DEFAULT_ROW_LIMIT}) unless the query is a single-row aggregate. "
        f"Never write INSERT/UPDATE/DELETE/DROP/ALTER — SELECT only. "
        f"If the user's intent was clarified below, honor that clarification exactly "
        f"in your ORDER BY / aggregation choice. Only reference columns that actually "
        f"appear in the schema below — never invent a column name (e.g. a rank or "
        f"score column) unless you compute it yourself with a window function or "
        f"expression in the same query."
    )
    hint_block = f"\nClarified intent: {resolved_hint}\n" if resolved_hint else ""
    error_block = ""
    if previous_error:
        error_block = (
            f"\nYour previous attempt failed when actually run against the database:\n"
            f"Previous SQL: {previous_sql}\n"
            f"Database error: {previous_error}\n"
            f"Fix the query so it only references real columns from the schema below.\n"
        )
    user = f"Schema:\n{schema.as_prompt_block()}\n{hint_block}{error_block}\nQuestion: {query}"

    return llm_router.call_structured(
        provider=provider,
        api_key=api_key,
        system_prompt=system,
        user_prompt=user,
        schema=SQLGenerationResult,
    )


# ---------------------------------------------------------------------------
# 3. Validation (defense in depth — never trust model output directly)
# ---------------------------------------------------------------------------

def validate_sql(sql: str, dialect: str = "duckdb") -> ValidationResult:
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as e:
        return ValidationResult(is_safe=False, reason=f"SQL failed to parse: {e}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return ValidationResult(is_safe=False, reason="Only a single SQL statement is allowed.")

    stmt = statements[0]

    if not isinstance(stmt, exp.Select):
        return ValidationResult(
            is_safe=False,
            reason="Only SELECT statements are allowed (no INSERT/UPDATE/DELETE/DDL).",
        )

    forbidden_types = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create)
    for node in stmt.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, forbidden_types):
            return ValidationResult(is_safe=False, reason=f"Disallowed statement type: {type(n).__name__}")

    cleaned = stmt
    if not cleaned.args.get("limit"):
        cleaned = cleaned.limit(DEFAULT_ROW_LIMIT)

    return ValidationResult(is_safe=True, cleaned_sql=cleaned.sql(dialect=dialect))
