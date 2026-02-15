import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

class AgentState(TypedDict):
    # This keeps track of the conversation on your Mac's memory
    messages: Annotated[list[BaseMessage], "The chat history"]

model = ChatAnthropic(model="claude-3-5-sonnet-latest")

def call_claude(state: AgentState):
    response = model.invoke(state['messages'])
    return {"messages": [response]}

# Define the Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", call_claude)
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)

app = workflow.compile()