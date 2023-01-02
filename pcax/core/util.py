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


def make_args(f: Callable, args=(), kwargs={}) -> List[str]:
    args_list = []
    args_it = iter(args)

    for parameter in inspect.signature(f).parameters.values():
        if parameter.name in kwargs:
            args_list.append(kwargs[parameter.name])
        else:
            try:
                args_list.append(next(args_it))
            except StopIteration:
                args_list.append(parameter.default)

    return args_list


def repr_function(f: Callable) -> str:
    """Human readable function representation."""
    signature = inspect.signature(f)
    args = [f'{k}={v.default}' for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty]
    args = ', '.join(args)
    while not hasattr(f, '__name__'):
        if not hasattr(f, 'func'):
            break
        f = f.func
    if not hasattr(f, '__name__') and hasattr(f, '__class__'):
        return f.__class__.__name__
    if args:
        return f'{f.__name__}(*, {args})'
    return f.__name__
