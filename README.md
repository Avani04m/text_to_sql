# Text-to-SQL with Clarification Engineering

A Streamlit application that converts natural-language questions into SQL using **Groq LLMs**, while handling ambiguity before generating the query.

Instead of silently assuming what a user means by terms such as **"best customer"**, **"most valuable product"**, or **"recent orders"**, the system detects possible interpretations and asks the user to clarify their intent through clickable options.

The selected interpretation is then used to generate, validate, and execute SQL against a DuckDB database.

---

## Why this project exists

Natural-language questions are often ambiguous.

For example:

> "Who is the best customer?"

There are several valid interpretations:

- Customer with the **highest total spending**
- Customer with the **most orders**
- Customer with the **highest total quantity purchased**
- Customer with the **highest average order value**
- Customer with the **fewest returns**

A traditional Text-to-SQL system may silently select one interpretation and generate SQL.

This project takes a different approach:

```text
User Question
      │
      ▼
Ambiguity Detection
      │
      ├── Not ambiguous ───────────────┐
      │                                │
      ▼                                │
Clarification Options                  │
      │                                │
      ▼                                │
User selects interpretation            │
      │                                │
      └────────────────────────────────┘
                       │
                       ▼
              SQL Generation
                       │
                       ▼
               SQL Validation
                       │
                       ▼
                 DuckDB
                       │
                       ▼
                Query Result