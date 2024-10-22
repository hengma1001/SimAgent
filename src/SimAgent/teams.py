from functools import partial

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .agent import CodingTeamState, MDTeamState, create_supervisor, worker_node
from .tools import python_repl, shell_tool, tavily_tool, write_script


def get_llm(model_name: str):
    return ChatOpenAI(model=model_name)


class Biophys_workflow:
    def __init__(self, llm_model="gpt-4o-mini") -> None:
        self.llm = get_llm(llm_model)

    def build_graph(self):
        search_agent = worker_node(self.llm, [tavily_tool], name="Search")
        preprocess_agent = worker_node(self.llm, [write_script], name="Preprocessor")
        md_agent = worker_node(self.llm, [python_repl], name="Simulator", messages_modifier="")
        analysis_agent = worker_node(self.llm, [python_repl], name="Analysis")
        supervisor_agent = create_supervisor(
            self.llm,
            "You are a supervisor managing a molecular dynamics simulation team,"
            " with following workers: Search, Preprocessor, Simulator and Analysis."
            " Given the following user request,"
            " respond with the worker to act next. Each worker will perform a task"
            " and repond with their results and status. When finished, respond with"
            " FINISH. ",
            ["Search", "Preprocessor", "Simulator", "Analysis"],
        )

        md_graph = StateGraph(MDTeamState)
        md_graph.add_node("Search", search_agent)
        md_graph.add_node("Preprocessor", preprocess_agent)
        md_graph.add_node("Simulator", md_agent)
        md_graph.add_node("Analysis", analysis_agent)
        md_graph.add_node("supervisor", supervisor_agent)

        md_graph.add_edge("Search", "supervisor")
        md_graph.add_edge("Preprocessor", "supervisor")
        md_graph.add_edge("Simulator", "supervisor")
        md_graph.add_edge("Analysis", "supervisor")

        md_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "Search": "Search",
                "Preprocessor": "Preprocessor",
                "Simulator": "Simulator",
                "Analysis": "Analysis",
                "FINISH": END,
            },
        )

        md_graph.add_edge(START, "supervisor")
        chain = md_graph.compile()

        return chain


class CodingTeamWorkflow:

    def __init__(self, llm_model="gpt-4o-mini") -> None:
        self.llm = get_llm(llm_model)

    def build_graph(self):
        code_agent = worker_node(
            self.llm,
            [tavily_tool],
            name="coder",
            messages_modifier="""
            You are a python expert. \n
            Ensure any code you provide can be executed \n
            with all required imports, variables defined, \n
            proper docstrings, and type definition. \n
            Output only the code please. """,
        )
        # code_agent = partial(code_generate, coding_llm)
        review_agent = worker_node(
            self.llm,
            [python_repl],
            name="reviewer",
            messages_modifier="""
            You are a python expert. You need to criticize the code, and improve it. \n
            You will also run the code and report the errors. \n
            If the code itself has bugs or needs improvement, send it back to the coder. \n
            If the issue is the environment, output ENV. \n
            If the code is good, run it in current environment, and output DONE. """,
        )

        env_agent = worker_node(
            self.llm,
            [shell_tool],
            name="envBuilder",
            messages_modifier="""
            You are managing the system environment. You need to \n
            install the missing packages in the environment via pip or \n
            conda.
            """,
        )

        writing_agent = worker_node(self.llm, [write_script], name="writer")

        def router(state):
            # This is the router
            messages = state["messages"]
            last_message = messages[-1]
            # if last_message.tool_calls:
            #     # The previous agent is invoking a tool
            #     return "call_tool"
            if "DONE" in last_message.content:
                # Any agent decided the work is done
                return "DONE"
            if "FINISH" in last_message.content:
                # Any agent decided the work is done
                return "FINISH"
            return "coder"

        graph = StateGraph(CodingTeamState)
        graph.add_node("coder", code_agent)
        graph.add_node("reviewer", review_agent)
        graph.add_node("writer", writing_agent)
        graph.add_node("envBuilder", env_agent)

        graph.add_edge("coder", "reviewer")
        graph.add_edge("envBuilder", "reviewer")
        graph.add_conditional_edges("reviewer", router, {"coder": "coder", "DONE": "writer", "ENV": "envBuilder"})
        graph.add_edge("writer", END)

        graph.add_edge(START, "coder")

        return graph.compile()
