"""
Pydantic models shared across the pipeline.
Every LLM call in this project returns one of these — never raw parsed text.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    name: str
    type: str
    is_fk: bool = False
    references: Optional[str] = None  # "table.column"


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnInfo]
    description: Optional[str] = None
    row_count: Optional[int] = None

    def as_prompt_block(self) -> str:
        cols = ", ".join(
            f"{c.name} {c.type}"
            + (f" (possible FK -> {c.references})" if c.references else "")
            for c in self.columns
        )

        desc = f" -- {self.description}" if self.description else ""

        return f"{self.name}({cols}){desc}"


class DatabaseSchema(BaseModel):
    tables: list[TableSchema]

    def as_prompt_block(self) -> str:
        return "\n".join(t.as_prompt_block() for t in self.tables)


# ---------------------------------------------------------------------------
# Ambiguity detection / clarification
# ---------------------------------------------------------------------------

class ClarificationOption(BaseModel):
    label: str
    sql_hint: str
    column_ref: Optional[str] = None


class AmbiguityOption(BaseModel):
    """
    One schema-grounded interpretation of an ambiguous phrase.

    Example:
        label = "Highest total spending"
        sql_hint = "Rank customers by SUM(orders.total_amount) DESC"
        column_refs = ["orders.total_amount"]
    """
    label: str
    sql_hint: str
    column_refs: list[str]
    # Short explanation of why these columns make this interpretation
    # computable from the schema.
    schema_basis: str


class AmbiguityCheck(BaseModel):
    """
    Structured output from the dynamic schema-aware ambiguity detector.
    """

    is_ambiguous: bool

    ambiguous_terms: list[str] = Field(default_factory=list)

    question: Optional[str] = None

    options: list[AmbiguityOption] = Field(default_factory=list)

    reason: Optional[str] = None


class ClarificationRequest(BaseModel):
    original_query: str
    question: str
    options: list[ClarificationOption]

    source: Literal["schema_llm"] = "schema_llm"


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

class SQLGenerationResult(BaseModel):
    sql: str
    explanation: str
    tables_used: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class ValidationResult(BaseModel):
    is_safe: bool
    reason: Optional[str] = None
    cleaned_sql: Optional[str] = None


class PipelineResult(BaseModel):
    """Final object rendered by the UI for one turn."""
    original_query: str
    resolved_query: str
    sql: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
