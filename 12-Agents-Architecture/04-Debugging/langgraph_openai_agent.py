import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

model=ChatOpenAI(model="gpt-4o", temperature=0)


class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def make_default_graph():
    graph = StateGraph(State)

    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    # Add Nodes
    graph.add_node(node="agent", action=call_model)

    # Add Edges
    graph.add_edge(start_key=START, end_key="agent")
    graph.add_edge(start_key="agent", end_key=END)

    agent = graph.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph = StateGraph(State)

    # Add Nodes
    graph.add_node(node="agent", action=call_model)
    graph.add_node(node="tools", action=tool_node)
    
    # Add Edges
    graph.add_edge(start_key=START, end_key="agent")
    graph.add_conditional_edges(source="agent", path=should_continue)
    graph.add_edge(start_key="tools", end_key="agent")

    agent = graph.compile()
    return agent


# agent = make_default_graph()
agent = make_alternative_graph()