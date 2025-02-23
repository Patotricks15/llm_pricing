import sqlite3
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

DB_PATH = "/home/patrick/llm_pricing/example.db"

def create_result_tables():
    """
    Creates 3 tables (if they do not exist) to store elasticities:
      1. computed_product_elasticities
      2. computed_customer_elasticities
      3. computed_c_p_elasticities
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for elasticity by product
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_product_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    # Table for elasticity by customer
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_customer_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    # Table for elasticity by (customer, product)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS computed_c_p_elasticities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            product_id INTEGER,
            price_type TEXT,
            elasticity REAL
        );
    """)

    conn.commit()
    conn.close()

def compute_price_elasticity(df: pd.DataFrame, price_col: str) -> float:
    """
    Given a DataFrame with columns:
      - 'quantity'
      - <price_col> (e.g., 'regular_price' or 'sale_price')
    performs a log-log regression to obtain the elasticity:
        ln(quantity) = alpha + beta ln(price_col)
    Returns the value of 'beta' (the elasticity).
    Raises ValueError if there are not enough data points.
    """
    # Remove invalid rows (quantity <= 0 or price <= 0)
    df = df[(df["quantity"] > 0) & (df[price_col] > 0)].copy()
    if len(df) < 2:
        raise ValueError("There are not enough data points to calculate elasticity (need >=2 points).")

    df["log_quantity"] = df["quantity"].apply(math.log)
    df["log_price"] = df[price_col].apply(math.log)

    # OLS regression using formula
    model = smf.ols("log_quantity ~ log_price", data=df).fit()
    elasticity = model.params["log_price"]
    return np.abs(elasticity)

def compute_elasticities():
    """
    1. Reads the entire 'orders' table from the DB.
    2. Calculates elasticity for each product_id, each customer_id, and (customer_id, product_id).
    3. Saves the results into 3 separate tables.
    """

    # Connect and read data
    conn = sqlite3.connect(DB_PATH)
    df_orders = pd.read_sql_query("SELECT * FROM orders;", conn)
    conn.close()

    # 1) Identify distinct IDs
    product_ids = df_orders["product_id"].dropna().unique().tolist()
    customer_ids = df_orders["customer_id"].dropna().unique().tolist()

    # Create or clear the result tables
    create_result_tables()

    # Open connection for inserting results
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Elasticity by product
    price_types = ["regular_price", "sale_price"]  # corresponds to the columns in the DB
    for pid in product_ids:
        # Filter the DataFrame for only this product
        df_prod = df_orders[df_orders["product_id"] == pid]

        for pt in price_types:
            try:
                elasticity_val = compute_price_elasticity(df_prod, pt)
            except ValueError:
                # If there are not enough data points, ignore or save as NULL
                elasticity_val = None

            # Insert into the table
            cursor.execute("""
                INSERT INTO computed_product_elasticities (product_id, price_type, elasticity)
                VALUES (?, ?, ?);
            """, (pid, pt, abs(elasticity_val)))

    # Elasticity by customer
    for cid in customer_ids:
        # Filter the DataFrame for only this customer
        df_cust = df_orders[df_orders["customer_id"] == cid]

        for pt in price_types:
            try:
                elasticity_val = np.abs(compute_price_elasticity(df_cust, pt))
            except ValueError:
                elasticity_val = None

            cursor.execute("""
                INSERT INTO computed_customer_elasticities (customer_id, price_type, elasticity)
                VALUES (?, ?, ?);
            """, (cid, pt, abs(elasticity_val)))

    # Elasticity by (customer, product)

    # Instead of generating all combinations, we extract directly from df_orders
    # Alternatively, we could merge sets, but it's better to use groupby and iterate
    df_cust_prod = df_orders[["customer_id","product_id"]].drop_duplicates()
    # df_cust_prod has unique pairs (customer_id, product_id)

    for _, row in df_cust_prod.iterrows():
        cid = row["customer_id"]
        pid = row["product_id"]
        # Filter
        df_sub = df_orders[(df_orders["customer_id"] == cid) & (df_orders["product_id"] == pid)]
        print(df_sub)

        for pt in price_types:
            try:
                elasticity_val = compute_price_elasticity(df_sub, pt)
            except ValueError:
                elasticity_val = None

            cursor.execute("""
                INSERT INTO computed_c_p_elasticities (customer_id, product_id, price_type, elasticity)
                VALUES (?, ?, ?, ?);
            """, (cid, pid, pt, abs(elasticity_val)))
    # Commit and close connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    compute_elasticities()
    print("Elasticities calculated and stored in the tables:")
    print(" - computed_product_elasticities")
    print(" - computed_customer_elasticities")
    print(" - computed_c_p_elasticities")
