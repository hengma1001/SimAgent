import operator
from functools import partial
from typing import Annotated, List, Sequence

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from pydantic import Field
from typing_extensions import TypedDict


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


# ResearchTeam graph state
class MDTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


# ResearchTeam graph state
class CodingTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}


def worker_node(llm, tools, name, **kwargs):
    agent = create_react_agent(llm, tools=tools, **kwargs)
    node = partial(agent_node, agent=agent, name=name)
    return node


def create_supervisor(llm, system_prompt, members) -> str:
    """An LLM-based router."""

    trimmer = trim_messages(
        max_tokens=100000,
        strategy="last",
        token_counter=llm,
        include_system=True,
    )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | trimmer
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )


# def code_review_agent(state: CodingTeamState):
#     """Run the python code

#     Parameters
#     ----------
#     state : CodingTeamState
#         _description_
#     """
#     code = state.messages[-1].content

#     if code == "":
#         error_message = [BaseMessage(content=f"Missing entry in the code section. ", type="text")]
#         state.messages += error_message
#         state.next = "coder"
#         return {"review": state}

#     try:
#         exec(code)
#         error_message = [BaseMessage(content=f"Run success! ", type="text")]
#         state.messages += error_message
#         state.next = "DONE"
#     except Exception as e:
#         error_message = [BaseMessage(content=f"Your solution failed the test: {e}", type="text")]
#         state.messages += error_message
#         state.next = "coder"

#     return {"review": state}


# def code_generate(llm, state: CodingTeamState):
#     """_summary_

#     Parameters
#     ----------
#     llm : _type_
#         llm model for code generation
#     state : CodingTeamState
#         _description_
#     """
#     messages = state.messages
#     # code = state.code

#     code_gen_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """ You are a coding assistant with expertise in Python. \n
#                 Ensure any code you provide can be executed with all required imports and variables \n
#                 defined. Please only output the code. \n Here is the user question: """,
#             ),
#             ("placeholder", "{messages}"),
#         ]
#     )

#     chain = code_gen_prompt | llm

#     code_solution = chain.invoke({"messages": messages})

#     code_solution.next = "review"

#     return {"coder": code_solution}
