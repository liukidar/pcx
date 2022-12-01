from typing import Callable, List
import inspect


def positional_args_names(f: Callable) -> List[str]:
    """Returns the ordered names of the positional arguments of a function."""
    return list(
        p.name
        for p in inspect.signature(f).parameters.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
