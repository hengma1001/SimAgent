from pathlib import Path
from typing import Annotated

from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

repl = PythonREPL()
shell = ShellTool()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def write_script(
    content: Annotated[str, "Python script content to be written into a document. "],
    file_name: Annotated[str, "File path to save the python script. "],
) -> Annotated[str, "Path of the saved document file."]:
    """
    Create and save python script to the current working directory
    """
    with open(file_name, "w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def shell_tool(cmd: Annotated[str, "The shell command to build the conda environment"]):
    """
    Use this to execute the shell command, Mainly to build conda environment

    Parameters
    ----------
    cmd : Annotated[str, &quot;The shell command to build the conda environment&quot;]
        _description_
    """
    try:
        result = shell.run(cmd)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{cmd}\n```\nStdout: {result}"


# @tool
# def exec_code():
#     pass


def code_check(state):
    pass
