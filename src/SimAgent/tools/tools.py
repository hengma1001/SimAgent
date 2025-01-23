from pathlib import Path
from typing import Annotated

from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# def pdf_retriever(pdf_files: Annotated[str, "List of PDF file paths to extract text from."]):
#     """
#     Extract text from PDF files and add to vectorDB

#     Parameters
#     ----------
#     pdf_files : Annotated[str, &quot;List of PDF file paths to extract text from.&quot;]
#         _description_
#     """
#     docs = [WebBaseLoader(pdf_file).load() for pdf_file in pdf_files]
#     docs_list = [item for sublist in docs for item in sublist]

#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
#     doc_splits = text_splitter.split_documents(docs_list)

#     # Add to vectorDB
#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         collection_name="rag-chroma",
#         embedding=OpenAIEmbeddings(),
#     )
#     retriever = vectorstore.as_retriever()


def code_check(state):
    pass
