import operator
from functools import partial
from typing import Annotated, Any, List, Tuple, Union

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from .agent import SuperviseState, create_supervisor
from .node import parsl_tool_node
from .tools import (
    download_structure,
    fold_sequence,
    python_repl,
    run_enhance_md,
    run_simulation_ensemble,
    shell_tool,
    simulate_structure,
    tavily_tool,
    write_script,
)


def get_llm(model_name: str):
    return ChatOpenAI(model=model_name)


class base_team(BaseModel):
    llm_model: str = Field(default="gpt-4o-mini", description="the LLM model name")
    parsl_run: bool = Field(default=False, description="Whether to implement parsl for the tool node")
    human_in_loop: bool = Field(default=False, description="human approval at the end of the workflow")

    @property
    def llm(self):
        return get_llm(self.llm_model)

    def build_graph(self) -> Any:
        pass


class sim_team(base_team):
    hpc_run: bool = Field(default=False, description="Whether to run the simulation on HPC")

    def build_graph(self):
        if self.hpc_run:
            tools = [download_structure, run_simulation_ensemble]
            tool_node = ToolNode(tools)
        else:
            tools = [download_structure, simulate_structure]
            tool_node = parsl_tool_node(tools)

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


class research_sim_team(sim_team):
    n_search_results: int = 5

    def build_graph(self):

        # Get the prompt to use - you can modify this!
        # prompt = hub.pull("ih/ih-react-agent-executor")

        # Choose the LLM that will drive the agent
        search_agent = create_react_agent(
            self.llm,
            [TavilySearchResults(max_results=self.n_search_results)],
            state_modifier="You are a researcher. Given the following objective, search for the relevant information.",
        )

        # sim_tools = [download_structure, fold_sequence, simulate_structure]
        sim_agent = sim_team(llm_model=self.llm_model, parsl_run=self.parsl_run).build_graph()
        # create_react_agent(self.llm, sim_tools, state_modifier=prompt)

        supervisor_agent = create_supervisor(
            get_llm(self.llm_model),
            "You are a supervisor tasked to execute the plan with the"
            " following workers:  researcher and simulator."
            " Given the following conversation, respond with the worker to act next."
            " When finished, respond with FINISH.",
            ["researcher", "simulator"],
        )

        workflow = StateGraph(SuperviseState)

        # Add the plan node
        workflow.add_node("manager", supervisor_agent)

        # Add the execution step
        workflow.add_node("researcher", search_agent)
        workflow.add_node("simulator", sim_agent)

        workflow.add_edge(START, "manager")

        # From plan we go to the manager
        workflow.add_edge("researcher", "manager")
        workflow.add_edge("simulator", "manager")

        workflow.add_conditional_edges(
            "manager",
            # Next, we pass in the function that will determine which node is called next.
            lambda x: x["next"],
            {"simulator": "simulator", "researcher": "researcher", "FINISH": END},
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        app = workflow.compile()
        return app


class enhance_md_team(base_team):

    def build_graph(self):
        tools = [run_enhance_md]
        tool_node = ToolNode(tools)
        model = self.llm.bind_tools(tools)
        simulator_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a simulator tasked with running a molecular dynamics simulation. You will be running multiple iteraction of enchanced MD simulations to reach the target state.""",
                )
            ]
        )

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
