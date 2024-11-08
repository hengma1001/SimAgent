import operator
from functools import partial
from typing import Annotated, Any, List, Tuple, Union

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

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
    parsl_run: bool = Field(default=False, description="Whether to implement parsl for the tool node")

    def build_graph(self):
        tools = [tavily_tool]

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("ih/ih-react-agent-executor")
        prompt.pretty_print()

        # Choose the LLM that will drive the agent
        agent_executor = create_react_agent(self.llm, tools, state_modifier=prompt)

        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        planner = planner_prompt | self.llm.with_structured_output(Plan)

        replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
        This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        Your objective was this:
        {input}

        Your original plan was this:
        {plan}

        You have currently done the follow steps:
        {past_steps}

        Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
        )

        replanner = replanner_prompt | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Act)

        def execute_step(state: PlanExecute):
            plan = state["plan"]
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            task = plan[0]
            task_formatted = f"""For the following plan:
        {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
            agent_response = agent_executor.invoke({"messages": [("user", task_formatted)]})
            return {
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        def plan_step(state: PlanExecute):
            plan = planner.invoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps}

        def replan_step(state: PlanExecute):
            output = replanner.invoke(state)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        def should_end(state: PlanExecute):
            if "response" in state and state["response"]:
                return END
            else:
                return "agent"

        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", plan_step)

        # Add the execution step
        workflow.add_node("agent", execute_step)

        # Add a replan node
        workflow.add_node("replan", replan_step)

        workflow.add_edge(START, "planner")

        # From plan we go to agent
        workflow.add_edge("planner", "agent")

        # From agent, we replan
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            should_end,
            ["agent", END],
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        app = workflow.compile()
        return app


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(description="different steps to follow, should be in sorted order")


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
