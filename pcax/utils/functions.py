import inspect


def ensure_list(x, even_if_none=False):
    return x if (isinstance(x, list) or (x is None and not even_if_none)) else [x]


def ensure_tuple(x, even_if_none=False):
    return x if (isinstance(x, tuple) or (x is None and not even_if_none)) else [x]


def all_kwargs(fn, *args, get_params_names: bool = False, **kwargs):
    parameters = inspect.signature(fn).parameters
    it_args = iter(args)

    # Get the names of the remaining arguments of the function
    r_kwargs = {}
    for name in parameters:
        try:
            r_kwargs[name] = next(it_args) if name not in kwargs else kwargs[name]
        except StopIteration:
            # r_kwargs[name] = parameters[name].default
            pass

    if get_params_names:
        return r_kwargs, parameters
    else:
        return r_kwargs


def call_kwargs(fn, *args, **kwargs):
    parameters = inspect.signature(fn).parameters
    parameters_names = tuple(
        filter(
            lambda param_name: parameters[param_name].kind
            in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ],
            parameters,
        )
    )

    return fn(
        *args, **{name: kwargs[name] for name in parameters_names if name in kwargs}
    )
