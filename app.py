from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.tools import DuckDuckGoSearchResults

duckduckgo_tool = DuckDuckGoSearchResults()

# Define the state type
class State(TypedDict):
    question: str
    sql_output: str
    final_output: str

# ---------------------------
# Create the SQLAgent
# ---------------------------
# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:////home/patrick/llm_pricing/example.db")
# Instantiate the LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Create the SQL toolkit and get its tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()


problem_chain = LLMMathChain.from_llm(llm=model)
math_tool = Tool.from_function(name="Calculator",
                func=problem_chain.run,
                description="Useful for when you need to answer questions  about math. This tool is only for math questions and nothing else. Only input math expressions.")




# Define a system prompt for the SQLAgent
sql_prefix = (
    "You are a SQLAgent specialized in interacting with a SQLite database. "
    "Given a question, generate a syntactically correct SQLite query that returns relevant information "
    "from the following tables:\n\n"
    "orders (retailer ID, store ID, customer ID, timestamp, product ID, quantity, regular price, sale price) \n"
    "products (retailer ID, store ID, product ID, product name, product description, category name, department name) \n"
    "computed_product_elasticities (product ID, price type, elasticity) \n"
    "computed_customer_elasticities (customer ID, price type, elasticity) \n"
    "computed_c_p_elasticities (customer ID, product ID, price type, elasticity) \n\n"
    "Do not perform any DML statements. Return the query results as a concise text output."
)
sql_system_message = SystemMessage(content=sql_prefix)
# Create the SQL agent
sql_agent = create_react_agent(model, sql_tools, messages_modifier=sql_system_message)

# ---------------------------
# Create the PricingAnalystAgent
# ---------------------------
pricing_prefix = (
    "You are a PricingAnalystAgent, an expert in pricing analysis and strategy. "
    "Given a question and context from a SQL query, provide a clear, concise final answer with insights and recommendations "
    "regarding pricing strategy. Do not generate SQL queries here; just analyze the provided context."
)
pricing_system_message = SystemMessage(content=pricing_prefix)
# Create the Pricing Analyst agent (no extra tools needed)
pricing_agent = create_react_agent(model, tools=[math_tool], messages_modifier=pricing_system_message)

# ---------------------------
# Build the State Graph
# ---------------------------
# The state graph has two nodes: "SQLAgent" and "PricingAnalyst".
# The flow is: START -> SQLAgent -> PricingAnalyst -> END
builder = StateGraph(State)

# Node for SQLAgent: takes the input question, invokes the SQL agent, and stores its output as 'sql_output'.
builder.add_node("SQLAgent", lambda state: {
    "sql_output": sql_agent.invoke({"messages": [HumanMessage(content=state["question"])]})
})

# Node for PricingAnalyst: takes the original question and the SQL agent's output, and produces the final answer.
builder.add_node("PricingAnalyst", lambda state: {
    "final_output": pricing_agent.invoke({
        "messages": [HumanMessage(content=f"Question: {state['question']}\nSQL Output: {state['sql_output']}")]
    })
})

builder.add_edge(START, "SQLAgent")
builder.add_edge("SQLAgent", "PricingAnalyst")
builder.add_edge("PricingAnalyst", END)

# Compile the state graph
graph = builder.compile()

png_bytes = graph.get_graph(xray=1).draw_mermaid_png()

# Save the PNG data to a file
with open("elasticity_graph.png", "wb") as f:
    f.write(png_bytes)
# ---------------------------
# REPL loop to ask questions and get final answers
# ---------------------------
while True:
    user_question = input("Enter your question: ")
    initial_state: State = {"question": user_question, "sql_output": "", "final_output": ""}
    final_state = graph.invoke(initial_state)
    print("Final Answer:", final_state["final_output"]['messages'][-1].content)
    print("----")
