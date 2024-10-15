from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .agent import CodingTeamState, MDTeamState, create_supervisor, worker_node
from .tools import python_repl, tavily_tool, write_script


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


# class CodingTeamWorkflow:

#     def __init__(self, llm_model="gpt-4o-mini") -> None:
#         self.llm = get_llm(llm_model)

#     def build_graph(self):
#         coding_agent = worker_node(self.llm, [write_script], name="Coder")
#         tool_node = ToolNode([python_repl])

#         graph = StateGraph(CodingTeamState)
#         graph.add_node("coder", coding_agent)
#         graph.add_node("tools", tool_node)
