from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from typing import Optional

# We’ll use statsmodels + pandas for OLS log-log regression
import math
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import text

# For the custom tool:
from langchain.tools.base import BaseTool

def run_query(db: SQLDatabase, query: str, params=None):
    """
    Helper function that executes a SQL query using the db's SQLAlchemy engine
    and returns the fetched rows as a list of tuples.
    """
    if params is None:
        params = ()
    with db.engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()
    return rows
# ---------------------------------------
# 1. Database Setup & Model Initialization
# ---------------------------------------
db = SQLDatabase.from_uri("sqlite:////home/patrick/llm_pricing/example.db")

model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# ---------------------------------------
# 2. The Custom Elasticity Tool
# ---------------------------------------
class ElasticityTool(BaseTool):
    """
    A custom tool to compute price elasticity from data in the 'orders' table.
    Uses a log-log OLS regression to estimate elasticity.
    """

    # Pydantic requires type annotations for fields
    name: str = "compute_elasticity"
    description: str = (
        "Use this tool to compute the elasticity for a given product, or for a (product, customer) pair. "
        "It runs a log-log OLS regression on the data from the 'orders' table. "
        "Required arguments:\n"
        " - product_id (int), optional if you only want a customer-based elasticity\n"
        " - customer_id (int), optional if you only want a product-based elasticity\n"
        " - price_type (str): 'regular' or 'sale' (default 'regular')\n"
        "If both product_id and customer_id are given, it calculates elasticity for that combination.\n"
        "If only product_id is given, it calculates elasticity across all orders of that product.\n"
        "If only customer_id is given, it calculates elasticity for that customer across all products.\n"
    )

    # We'll need a reference to the DB, so we can pass it in at construction time
    def __init__(self, db: SQLDatabase, **kwargs):
        super().__init__(**kwargs)
        self.db = db

    def _run(
        self,
        product_id: Optional[int] = None,
        customer_id: Optional[int] = None,
        price_type: str = "regular"
    ) -> str:
        """
        Synchronous method to query data and compute elasticity.
        Returns a string describing the elasticity result (or an error).
        """
        # Validate price_type
        if price_type not in ("regular", "sale"):
            return "Error: price_type must be 'regular' or 'sale'."

        price_col = "regular_price" if price_type == "regular" else "sale_price"

        # Build WHERE conditions
        conditions = []
        params = []
        if product_id is not None:
            conditions.append("product_id = ?")
            params.append(product_id)
        if customer_id is not None:
            conditions.append("customer_id = ?")
            params.append(customer_id)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Query the needed data
        query = f"""
            SELECT quantity, {price_col} AS price
            FROM orders
            {where_clause}
        """

        rows = run_query(self.db, query, params)
        if not rows:
            return "No matching data found to compute elasticity."

        # Convert to a pandas DataFrame
        df = pd.DataFrame(rows, columns=["quantity", "price"])

        # Filter out invalid rows
        df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()
        if len(df) < 2:
            return "Not enough data points to compute elasticity (need >= 2)."

        # Log transform
        df["log_qty"] = df["quantity"].apply(math.log)
        df["log_price"] = df["price"].apply(math.log)

        # OLS regression with statsmodels
        model_ols = smf.ols("log_qty ~ log_price", data=df).fit()
        elasticity = model_ols.params["log_price"]

        info = []
        if product_id is not None:
            info.append(f"product_id={product_id}")
        if customer_id is not None:
            info.append(f"customer_id={customer_id}")

        return (
            f"Elasticity for ({', '.join(info)}), price_type={price_type}: "
            f"{elasticity:.4f}"
        )

    async def _arun(self, *args, **kwargs):
        """
        Async version not used here; just call _run if you don't need truly async behavior.
        """
        return self._run(*args, **kwargs)
# ---------------------------------------
# 3. Build the Standard SQL Toolkit
# ---------------------------------------
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()  # The standard SQL query tools

# ---------------------------------------
# 4. Combine Tools: SQL Tools + Elasticity Tool
# ---------------------------------------
elasticity_tool = ElasticityTool(db=db)
tools = sql_tools + [elasticity_tool]

# ---------------------------------------
# 5. System Prompt for the Agent
# ---------------------------------------
SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.


You are an agent designed to interact with a SQL database and a custom elasticity tool.
- You have SQL tools to query the database schema and run queries.
- You also have a 'compute_elasticity' tool that can compute price elasticity.
- If the user asks for an elasticity, you should use 'compute_elasticity' with the appropriate arguments.
- If the user wants direct data from the DB, use the SQL tools.

IMPORTANT:
- 'compute_elasticity' expects arguments like product_id, customer_id, and price_type.
- If only product_id is given, it will compute elasticity for that product across all customers.
- If only customer_id is given, it will compute elasticity for that customer across all products.
- If both are given, it computes elasticity for that combo.
- By default, use price_type='regular' unless user explicitly says 'sale price'.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.)."""

system_message = SystemMessage(content=SQL_PREFIX)

# ---------------------------------------
# 6. Create the ReAct Agent
# ---------------------------------------
agent_executor = create_react_agent(
    model=model,
    tools=tools,
    messages_modifier=system_message
)

# ---------------------------------------
# 7. Simple Chat Loop
# ---------------------------------------
while True:
    query = input("Text:")
    for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}
    ):
        print(s[list(s.keys())[0]]['messages'][-1].content)
        print("----")