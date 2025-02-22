import math
import sqlite3
from typing import Optional

import pandas as pd
import statsmodels.formula.api as smf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ------------------------------------------------------------------
# 1. FastAPI Initialization
# ------------------------------------------------------------------

app = FastAPI(
    title="Price Elasticity API",
    description="""API that computes regular_price or sale_price elasticity 
                   for a product, a customer, or a specific customer-product combination.""",
    version="1.0.0",
)

DATABASE_PATH = "example.db"  # Adjust if using a different DB or path


# ------------------------------------------------------------------
# 2. Helper Functions: Query + Compute Elasticity
# ------------------------------------------------------------------

def fetch_data_from_db(query: str, params: tuple = ()) -> pd.DataFrame:
    """Fetch data from the SQLite database using a SQL query."""
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def compute_price_elasticity(df: pd.DataFrame, price_col: str) -> float:
    """
    Given a DataFrame with columns [quantity, price_col],
    compute log-log OLS to estimate elasticity:
        ln(quantity) = alpha + beta ln(price_col)
    where beta is the elasticity we want.
    
    :param df: DataFrame with columns "quantity" and price_col
    :param price_col: either "regular_price" or "sale_price"
    :return: elasticity (float)
    """
    # Filter out rows with zero or negative quantity/price
    df = df[(df["quantity"] > 0) & (df[price_col] > 0)].copy()
    
    if len(df) < 2:
        raise ValueError("Not enough valid data points to run the regression (need at least 2).")
    
    # Create log columns
    df["log_quantity"] = df["quantity"].apply(math.log)
    df["log_price"] = df[price_col].apply(math.log)

    # OLS regression using statsmodels
    model = smf.ols("log_quantity ~ log_price", data=df).fit()
    elasticity = model.params["log_price"]
    
    return elasticity


# ------------------------------------------------------------------
# 3. Pydantic Models for Input (Optional for GET requests)
#    We'll demonstrate GET endpoints with query params, but you can
#    also implement POST if you'd prefer sending a JSON body.
# ------------------------------------------------------------------

class ProductElasticityRequest(BaseModel):
    product_id: int
    price_type: str = "regular"  # or "sale"

class CustomerElasticityRequest(BaseModel):
    customer_id: int
    price_type: str = "regular"

class CustomerProductElasticityRequest(BaseModel):
    customer_id: int
    product_id: int
    price_type: str = "regular"


# ------------------------------------------------------------------
# 4. Endpoints
# ------------------------------------------------------------------

@app.get("/elasticity/product")
def get_product_elasticity(product_id: int, price_type: str = "regular"):
    """
    Computes elasticity for a single product (across all orders).
    price_type can be 'regular' or 'sale'.
    
    Example:
      GET /elasticity/product?product_id=123&price_type=sale
    """
    if price_type not in ["regular", "sale"]:
        raise HTTPException(status_code=400, detail="price_type must be 'regular' or 'sale'")
    
    # Map price_type to actual column in DB
    price_col = "regular_price" if price_type == "regular" else "sale_price"
    
    # 1. Fetch data for the given product_id
    query = """
        SELECT quantity, {} as price
        FROM orders
        WHERE product_id = ?
    """.format(price_col)
    
    df = fetch_data_from_db(query, (product_id,))
    
    # 2. Compute elasticity
    try:
        elasticity_value = compute_price_elasticity(df, "price")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "type": "product_elasticity",
        "product_id": product_id,
        "price_type": price_type,
        "elasticity": elasticity_value
    }


@app.get("/elasticity/customer")
def get_customer_elasticity(customer_id: int, price_type: str = "regular"):
    """
    Computes elasticity for a single customer across ALL products they purchased.
    
    Example:
      GET /elasticity/customer?customer_id=789&price_type=regular
    """
    if price_type not in ["regular", "sale"]:
        raise HTTPException(status_code=400, detail="price_type must be 'regular' or 'sale'")
    
    price_col = "regular_price" if price_type == "regular" else "sale_price"
    
    # 1. Fetch data for the given customer_id
    query = """
        SELECT quantity, {} as price
        FROM orders
        WHERE customer_id = ?
    """.format(price_col)
    
    df = fetch_data_from_db(query, (customer_id,))
    
    # 2. Compute elasticity
    try:
        elasticity_value = compute_price_elasticity(df, "price")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "type": "customer_elasticity",
        "customer_id": customer_id,
        "price_type": price_type,
        "elasticity": elasticity_value
    }


@app.get("/elasticity/customer_product")
def get_customer_product_elasticity(customer_id: int, product_id: int, price_type: str = "regular"):
    """
    Computes elasticity for a single customer + single product combination.
    
    Example:
      GET /elasticity/customer_product?customer_id=789&product_id=123&price_type=sale
    """
    if price_type not in ["regular", "sale"]:
        raise HTTPException(status_code=400, detail="price_type must be 'regular' or 'sale'")
    
    price_col = "regular_price" if price_type == "regular" else "sale_price"
    
    # 1. Fetch data for the given customer_id and product_id
    query = f"""
        SELECT quantity, {price_col} as price
        FROM orders
        WHERE customer_id = ?
          AND product_id = ?
    """
    df = fetch_data_from_db(query, (customer_id, product_id))
    
    # 2. Compute elasticity
    try:
        elasticity_value = compute_price_elasticity(df, "price")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "type": "customer_product_elasticity",
        "customer_id": customer_id,
        "product_id": product_id,
        "price_type": price_type,
        "elasticity": elasticity_value
    }
