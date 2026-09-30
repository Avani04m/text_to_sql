# Text-to-SQL with Clarification Engineering

A Streamlit app that turns natural-language questions into SQL — but instead
of silently guessing on ambiguous business terms ("best customer", "top
product", "recent"), it detects the ambiguity and asks a clarifying question
first, via clickable options, before generating any SQL.

## Why this exists

"Who was the best customer last month?" has at least three valid, mutually
exclusive SQL interpretations:
- The customer who **spent the most money**
- The customer who **placed the most orders**
- The customer who **returned the fewest items**

Most text-to-SQL demos just pick one silently (usually whichever the LLM's
training data biases it toward) and give you a confident, wrong-for-your-intent
answer. This project detects that ambiguity up front and asks.

## Architecture

```
User question
     │
     ▼
[1] Ambiguity check  ──ambiguous──►  [2] Clarification buttons ──user picks──┐
     │ not ambiguous                                                         │
     ▼                                                                       │
[3] Schema-aware SQL generation  ◄────────────────────────────────────────────┘
     │
     ▼
[4] Safety validation (sqlglot: SELECT-only, single statement, LIMIT enforced)
     │
     ▼
[5] Execute (DuckDB or read-only Postgres) → render table
```

- **`models.py`** — every LLM call returns one of these Pydantic models. No
  raw text parsing anywhere in the pipeline.
- **`glossary.py`** — deterministic, zero-latency ambiguity detection for
  known business terms (fast path).
- **`llm_router.py`** — unified interface over Groq / OpenAI / Gemini, all
  forced into structured JSON output validated against a Pydantic schema,
  with one retry on parse failure.
- **`sql_pipeline.py`** — `check_ambiguity()` (glossary → LLM fallback),
  `generate_sql()`, `validate_sql()`.
- **`db.py`** — `DuckDBEngine` (default, in-memory, zero setup, seeded with
  a demo e-commerce dataset) and `PostgresEngine` (optional, for a real
  Postgres instance). CSV/XLSX uploads always go through DuckDB.
- **`streamlit_app.py`** — the UI: data source picker, model picker, chat
  loop with clarification buttons via `st.session_state`.

## Two-tier ambiguity detection

1. **Glossary match** (`glossary.py`) — instant, deterministic, no LLM call,
   no hallucination risk. Covers the terms you've explicitly thought about:
   best/worst customer, top product, recent, growth, most loyal.
2. **LLM fallback** — for questions that don't hit the glossary but might
   still be ambiguous (e.g. an uploaded dataset the glossary knows nothing
   about). A fast/cheap model (Groq by default) classifies `is_ambiguous`
   and proposes options, structured via `AmbiguityCheck`.

This distinction matters most for **uploaded datasets**: the glossary is
tuned to the demo e-commerce schema's semantics, so for unknown uploaded
schemas the LLM fallback carries all the weight.

## Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in at least one of GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
streamlit run streamlit_app.py
```

You can also skip `secrets.toml` and paste API keys directly into the
sidebar at runtime — either works.

## Deploying on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `streamlit_app.py`.
3. In the app's **Settings → Secrets**, paste the contents of your
   `secrets.toml` (API keys, and `DATABASE_URL` only if you're using the
   Postgres mode with a real hosted DB — Supabase/Neon/Railway all have
   free tiers).
4. Deploy. The demo mode (DuckDB, seeded data) works immediately with no
   external database needed.

## Safety notes

- The SQL validator (`sql_pipeline.validate_sql`) rejects anything that
  isn't a single `SELECT` statement — no `INSERT`/`UPDATE`/`DELETE`/DDL can
  reach the execution layer, regardless of what the model generates.
- A `LIMIT` is enforced on every query if the model didn't include one.
- If you connect a real Postgres database, **always use a read-only
  database role** for `DATABASE_URL` — the validator is defense-in-depth,
  not a substitute for DB-level permissions.
- CSV/XLSX uploads are sandboxed in an in-memory DuckDB instance per
  session — they never touch a shared/persistent database.

## Extending the glossary

Add new ambiguous terms to `GLOSSARY` in `glossary.py`:

```python
"healthy account": AmbiguousTerm(
    term="healthy account",
    question="What makes an account 'healthy' here?",
    options=[
        ClarificationOption(label="Active in the last 30 days", sql_hint="..."),
        ClarificationOption(label="No open support tickets", sql_hint="..."),
    ],
),
```

Matching is a case-insensitive, word-boundary substring match — no fuzzy
matching or embeddings needed for a glossary this size.

## Demo dataset

Seeded via `db.seed_demo_data()`: `customers`, `products`, `orders`,
`order_items`, `returns` — deliberately generated so that "most orders",
"most money spent", and "fewest returns" pick **different** top customers.
Try asking "who is the best customer" and compare each clarification option's
answer to see why the question needed asking in the first place.
