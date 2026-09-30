"""
Text-to-SQL with Clarification Engineering — Streamlit app.
Two data source modes:
  - Demo database (seeded DuckDB e-commerce dataset)
  - Upload your own CSV/XLSX (loaded into DuckDB)
"""
import streamlit as st
import pandas as pd

from db import DuckDBEngine, seed_demo_data, load_uploaded_file
from sql_pipeline import check_ambiguity, generate_sql, validate_sql
from models import ClarificationRequest

st.set_page_config(page_title="Text-to-SQL + Clarification", layout="wide")
# Session state init
def _init_state():
    defaults = {
        "messages": [],                 # chat history: list of dicts
        "pending_clarification": None,  # ClarificationRequest | None
        "engine": None,
        "data_source_label": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()
def _get_secret(key: str, default: str = "") -> str:
    """st.secrets.get() raises if secrets.toml doesn't exist at all — this makes it optional."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


@st.cache_resource
def get_demo_engine() -> DuckDBEngine:
    engine = DuckDBEngine()
    seed_demo_data(engine)
    return engine

# Sidebar: data source + model config

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Data source")
    source_mode = st.radio(
        "Choose a data source",
        ["Demo e-commerce database", "Upload CSV / XLSX"],
        label_visibility="collapsed",
    )

    if source_mode == "Demo e-commerce database":
        st.session_state.engine = get_demo_engine()
        st.session_state.data_source_label = "Demo DB (customers, orders, order_items, products, returns)"

    elif source_mode == "Upload CSV / XLSX":
        if not isinstance(st.session_state.engine, DuckDBEngine) or st.session_state.data_source_label is None \
                or "Uploaded" not in (st.session_state.data_source_label or ""):
            st.session_state.engine = DuckDBEngine()
        uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            table_name, n_rows = load_uploaded_file(st.session_state.engine, uploaded)
            st.session_state.data_source_label = f"Uploaded table `{table_name}` ({n_rows} rows)"
            st.success(f"Loaded {n_rows} rows into `{table_name}`")
        more = st.file_uploader("Add another table (optional)", type=["csv", "xlsx", "xls"], key="second_upload")
        if more is not None:
            table_name, n_rows = load_uploaded_file(st.session_state.engine, more)
            st.success(f"Loaded {n_rows} rows into `{table_name}`")


    st.divider()
    st.subheader("2. Model")
    provider_ambiguity = "groq"
    provider_sql = "groq"
    groq_api_key = _get_secret("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input(
        "Groq API key",
        value="",
        type="password",
        key="key_groq"
    )

    api_keys = {
    "groq": groq_api_key
    }

    st.caption(
    "Using Groq for schema-aware ambiguity detection "
    "and SQL generation."
)

    

    st.divider()
    if st.session_state.engine is not None:
        schema = st.session_state.engine.get_schema()
        with st.expander(f"📋 Schema — {st.session_state.data_source_label}", expanded=False):
            for t in schema.tables:
                st.code(t.as_prompt_block(), language="text")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_clarification = None
        st.rerun()
# Main chat area

st.title("Text-to-SQL with Clarification Engineering")
st.caption(
    "Ask a question about your data. If it contains an ambiguous business term "
    "(e.g. 'best customer'), you'll be asked to clarify before any SQL runs."
)

if st.session_state.engine is None:
    st.info("Pick a data source in the sidebar to get started.")
    st.stop()

# --- Data preview: see exactly what's loaded before asking questions ---
with st.expander(f"Preview data — {st.session_state.data_source_label}", expanded=False):
    schema = st.session_state.engine.get_schema()
    if not schema.tables:
        st.caption("No tables loaded yet.")
    else:
        table_names = [t.name for t in schema.tables]
        preview_table = st.selectbox("Table", table_names, key="preview_table_select")
        try:
            preview_df = st.session_state.engine.execute(f'SELECT * FROM "{preview_table}" LIMIT 50')
            st.dataframe(preview_df, use_container_width=True)
            st.caption(f"Showing first {len(preview_df)} rows of `{preview_table}`.")
        except Exception as e:
            st.error(f"Could not preview `{preview_table}`: {e}")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True)


def run_pipeline(user_query: str, resolved_hint: str | None = None):
    schema = st.session_state.engine.get_schema()
    dialect = st.session_state.engine.dialect

    with st.chat_message("assistant"):
        MAX_ATTEMPTS = 2
        gen = None
        final_sql = None
        result_df = None
        last_exec_error = None
        prev_sql_for_retry = None

        with st.spinner("Generating SQL..."):
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    gen = generate_sql(
                        query=user_query,
                        schema=schema,
                        provider=provider_sql,
                        api_key=api_keys[provider_sql],
                        resolved_hint=resolved_hint,
                        dialect=dialect,
                        previous_error=last_exec_error,
                        previous_sql=prev_sql_for_retry,
                    )
                except Exception as e:
                    st.error(f"SQL generation failed: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ SQL generation failed: {e}"}
                    )
                    return

                validation = validate_sql(gen.sql, dialect=dialect)
                if not validation.is_safe:
                    st.error(f"Blocked unsafe SQL: {validation.reason}")
                    st.code(gen.sql, language="sql")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Generated SQL was blocked by the validator: {validation.reason}",
                        "sql": gen.sql,
                    })
                    return

                final_sql = validation.cleaned_sql
                try:
                    result_df = st.session_state.engine.execute(final_sql)
                    last_exec_error = None
                    break  # success
                except Exception as e:
                    last_exec_error = str(e)
                    prev_sql_for_retry = final_sql
                    if attempt == MAX_ATTEMPTS:
                        st.error(f"Query execution failed after {MAX_ATTEMPTS} attempts: {e}")
                        st.code(final_sql, language="sql")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ Query execution failed: {e}",
                            "sql": final_sql,
                        })
                        return
                    # else: loop again, model gets the error and tries to self-correct

        st.markdown(gen.explanation)
        with st.expander("Generated SQL", expanded=False):
            st.code(final_sql, language="sql")
        st.dataframe(result_df, use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": gen.explanation,
            "sql": final_sql,
            "df": result_df,
        })


# --- pending clarification takes priority over new input ---
if st.session_state.pending_clarification:
    clar: ClarificationRequest = st.session_state.pending_clarification
    with st.chat_message("assistant"):
        st.markdown(f"**{clar.question}**")
        cols = st.columns(len(clar.options))
        for i, opt in enumerate(clar.options):
            if cols[i].button(opt.label, key=f"clar_opt_{i}_{opt.label}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"*(clarified: {opt.label})*",
                })
                resolved = clar.original_query
                st.session_state.pending_clarification = None
                run_pipeline(resolved, resolved_hint=opt.sql_hint)
                st.rerun()

user_query = st.chat_input(
    "e.g. Who was the best customer last month?",
    disabled=st.session_state.pending_clarification is not None,
)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    schema = st.session_state.engine.get_schema()
    clarification = check_ambiguity(
    query=user_query,
    schema=schema,
    provider=provider_ambiguity,
    api_key=api_keys.get(provider_ambiguity),
)

    if clarification:
        st.session_state.pending_clarification = clarification
        st.rerun()
    else:
        run_pipeline(user_query)