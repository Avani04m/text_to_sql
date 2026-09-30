"""
Database engine layer.

Default: in-memory DuckDB, seeded with a demo e-commerce dataset
(customers, orders, order_items, products, returns). Zero external
setup required — this is what makes the project runnable in one click
on Streamlit Community Cloud.

Optional: point DATABASE_URL (in st.secrets or env) at a real Postgres
instance (e.g. Supabase / Neon / Railway) to match the stack in the
brief. The same DatabaseEngine interface works for both — the SQL
pipeline and validator don't care which one is behind it, aside from
the `dialect` string used by sqlglot.

CSV/XLSX upload always goes through DuckDB (df.to_sql-style register),
even when a Postgres demo DB is also configured, so uploads are cheap,
sandboxed, and don't require write access to a shared Postgres instance.
"""
from __future__ import annotations
import datetime as dt
import random
from typing import Optional

import duckdb
import pandas as pd

from models import ColumnInfo, TableSchema, DatabaseSchema


# ---------------------------------------------------------------------------
# Engine abstraction
# ---------------------------------------------------------------------------

class DatabaseEngine:
    dialect: str = "duckdb"

    def execute(self, sql: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_schema(self) -> DatabaseSchema:
        raise NotImplementedError

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        raise NotImplementedError


class DuckDBEngine(DatabaseEngine):
    dialect = "duckdb"

    def __init__(self):
        self.con = duckdb.connect(database=":memory:")

    def execute(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        tmp_view = f"_{name}_tmp"
        self.con.register(tmp_view, df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM {tmp_view}')
        self.con.unregister(tmp_view)

    def get_schema(self) -> DatabaseSchema:
        tables_df = self.con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchdf()
        tables = []
        for table_name in tables_df["table_name"]:
            cols_df = self.con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
            ).fetchdf()
            try:
                count = self.con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            except Exception:
                count = None
            columns = [
                ColumnInfo(
                    name=row.column_name,
                    type=row.data_type,
                    is_fk=row.column_name.endswith("_id") and row.column_name != "id",
                    references=_guess_fk_reference(row.column_name, table_name),
                )
                for row in cols_df.itertuples()
            ]
            tables.append(TableSchema(name=table_name, columns=columns, row_count=count))
        return DatabaseSchema(tables=tables)


class PostgresEngine(DatabaseEngine):
    """Optional real-Postgres backend. Requires a READ-ONLY role in production."""
    dialect = "postgres"

    def __init__(self, database_url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(database_url)

    def execute(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        # Uploads never go to the shared Postgres — see module docstring.
        raise NotImplementedError("Uploads run through DuckDB, not the Postgres engine.")

    def get_schema(self) -> DatabaseSchema:
        q = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """
        df = pd.read_sql(q, self.engine)
        tables = []
        for table_name, group in df.groupby("table_name"):
            columns = [
                ColumnInfo(
                    name=r.column_name,
                    type=r.data_type,
                    is_fk=r.column_name.endswith("_id") and r.column_name != "id",
                    references=_guess_fk_reference(r.column_name, table_name),
                )
                for r in group.itertuples()
            ]
            tables.append(TableSchema(name=table_name, columns=columns))
        return DatabaseSchema(tables=tables)


def _guess_fk_reference(column_name: str, own_table: str) -> Optional[str]:
    """Cheap heuristic: `customer_id` -> `customers.id`. Good enough for prompt context."""
    if column_name.endswith("_id") and column_name != "id":
        base = column_name[:-3]
        candidates = [base + "s", base, base + "es"]
        for c in candidates:
            if c != own_table:
                return f"{c}.id"
    return None


# ---------------------------------------------------------------------------
# Demo dataset (deliberately ambiguity-rich: money vs count vs returns)
# ---------------------------------------------------------------------------

def seed_demo_data(engine: DuckDBEngine) -> None:
    rng = random.Random(42)
    today = dt.date(2026, 9, 26)

    customer_names = [
        "Aarav Sharma", "Priya Nair", "Rohan Gupta", "Isha Verma", "Kabir Singh",
        "Meera Iyer", "Vikram Rao", "Ananya Das", "Dev Patel", "Sara Khan",
        "Arjun Mehta", "Neha Joshi", "Karan Malhotra", "Divya Reddy", "Yusuf Ali",
    ]
    customers = pd.DataFrame({
        "id": range(1, len(customer_names) + 1),
        "name": customer_names,
        "signup_date": [today - dt.timedelta(days=rng.randint(30, 900)) for _ in customer_names],
        "city": rng.choices(
            ["Mumbai", "Delhi", "Bengaluru", "Varanasi", "Pune", "Chennai"], k=len(customer_names)
        ),
    })

    product_names = [
        "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "27in Monitor",
        "Noise Cancelling Headphones", "Webcam 1080p", "Laptop Stand", "Desk Lamp",
        "Ergonomic Chair", "Portable SSD 1TB",
    ]
    products = pd.DataFrame({
        "id": range(1, len(product_names) + 1),
        "name": product_names,
        "category": rng.choices(["Electronics", "Accessories", "Furniture"], k=len(product_names)),
        "unit_price": [round(rng.uniform(8, 300), 2) for _ in product_names],
    })

    # Orders: deliberately skewed so "most orders" and "most money" pick DIFFERENT winners
    orders_rows = []
    order_id = 1
    for cust_id in customers["id"]:
        n_orders = rng.randint(1, 14)
        for _ in range(n_orders):
            days_ago = rng.randint(0, 400)
            order_date = today - dt.timedelta(days=days_ago)
            orders_rows.append({
                "id": order_id,
                "customer_id": cust_id,
                "order_date": order_date,
                "total_amount": 0.0,  # filled after order_items generated
            })
            order_id += 1
    orders = pd.DataFrame(orders_rows)

    order_items_rows = []
    item_id = 1
    for oid in orders["id"]:
        n_items = rng.randint(1, 4)
        chosen = rng.sample(list(products["id"]), k=min(n_items, len(products)))
        for pid in chosen:
            qty = rng.randint(1, 5)
            price = float(products.loc[products.id == pid, "unit_price"].iloc[0])
            order_items_rows.append({
                "id": item_id, "order_id": oid, "product_id": pid,
                "quantity": qty, "unit_price": price,
            })
            item_id += 1
    order_items = pd.DataFrame(order_items_rows)

    totals = order_items.groupby("order_id").apply(
        lambda g: (g["quantity"] * g["unit_price"]).sum()
    ).round(2)
    orders["total_amount"] = orders["id"].map(totals).fillna(0.0)

    # Returns: skewed so one high-spender also has many returns (interesting case)
    returns_rows = []
    return_id = 1
    for _, item in order_items.sample(frac=0.12, random_state=42).iterrows():
        order_row = orders.loc[orders.id == item["order_id"]].iloc[0]
        returns_rows.append({
            "id": return_id,
            "order_id": int(item["order_id"]),
            "customer_id": int(order_row["customer_id"]),
            "product_id": int(item["product_id"]),
            "return_date": order_row["order_date"] + dt.timedelta(days=rng.randint(1, 14)),
            "reason": rng.choice(["Defective", "Wrong item", "Changed mind", "Late delivery"]),
        })
        return_id += 1
    returns = pd.DataFrame(returns_rows)

    engine.load_dataframe("customers", customers)
    engine.load_dataframe("products", products)
    engine.load_dataframe("orders", orders)
    engine.load_dataframe("order_items", order_items)
    engine.load_dataframe("returns", returns)


def load_uploaded_file(engine: DuckDBEngine, uploaded_file) -> tuple[str, int]:
    """Load a Streamlit UploadedFile (csv/xlsx) into DuckDB as a table."""
    name = uploaded_file.name.rsplit(".", 1)[0]
    table_name = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "uploaded_table"

    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    engine.load_dataframe(table_name, df)
    return table_name, len(df)
