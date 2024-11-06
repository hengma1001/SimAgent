from functools import partial
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, model_validator

from .agent import CodingTeamState, MDTeamState, create_supervisor, worker_node
from .node import parsl_tool_node
from .tools import (
    download_structure,
    fold_sequence,
    python_repl,
    shell_tool,
    simulate_structure,
    tavily_tool,
    write_script,
)


def get_llm(model_name: str):
    return ChatOpenAI(model=model_name)


class base_team(BaseModel):
    llm_model: str = Field(default="gpt-4o-mini", description="the LLM model name")
    human_in_loop: bool = Field(default=False, description="human approval at the end of the workflow")

    @property
    def llm(self):
        return get_llm(self.llm_model)

    def build_graph(self) -> Any:
        pass


class sim_team(base_team):
    parsl_run: bool = Field(default=False, description="Whether to implement parsl for the tool node")

    def build_graph(self):
        tools = [download_structure, fold_sequence, simulate_structure]
        if self.parsl_run:
            tool_node = parsl_tool_node(tools)
        else:
            tool_node = ToolNode(tools)
        model = self.llm.bind_tools(tools)

        def should_continue(state: MessagesState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "tools"
            return END

        def call_model(state: MessagesState):
            messages = state["messages"]
            response = model.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")

        app = workflow.compile()
        return app


class plan_exe_team(base_team):
    pass
