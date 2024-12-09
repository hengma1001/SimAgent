from typing import Any, Callable

from parsl.app.app import python_app


def parsl_compute_executor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for executing functions remotely using Parsl.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to execute remotely.

    Returns
    -------
    Callable[..., Any]
        A wrapped function that submits to Parsl Compute.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        app = python_app(func)
        parsl_inst = app(*args, *kwargs)
        return parsl_inst.result()

    return wrapper
