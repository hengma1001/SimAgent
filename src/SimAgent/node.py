from typing import Any, Optional

import parsl
from langchain_core.messages import ToolMessage
from langchain_core.runnables.base import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableLike,
    RunnableParallel,
    RunnableSequence,
)
from langchain_core.runnables.utils import Input
from langgraph.prebuilt import ToolNode
from parsl.app.app import python_app


class parsl_tool_node(ToolNode):
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        result = input["messages"]
        if self.func is None:
            raise TypeError(
                f'No synchronous function provided to "{self.name}".'
                "\nEither initialize with a synchronous function or invoke"
                " via the async API (ainvoke, astream, etc.)"
            )

        parsl_insts = []
        for tool_call in input["messages"][-1].tool_calls:
            tool_parsl = python_app(self.tools_by_name[tool_call["name"]].func)
            observation = tool_parsl(**tool_call["args"])
            parsl_insts.append([observation, tool_call["id"]])

        for inst, tool_id in parsl_insts:
            observation = inst.result()
            result.append(ToolMessage(content=observation, tool_call_id=tool_id))
        return {"messages": result}

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Any:
        pass
