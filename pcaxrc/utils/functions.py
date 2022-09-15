import inspect


def ensure_list(x, even_if_none=False):
    return x if (isinstance(x, list) or (x is None and not even_if_none)) else [x]


def all_kwargs(fn, *args, get_params_names: bool = False, **kwargs):
    parameters = inspect.signature(fn).parameters
    # Get the names of the arguments of the function corresponding to args
    parameters_names = tuple(
        filter(
            lambda param_name: parameters[param_name].kind
            in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ],
            parameters,
        )
    )[: len(args)]

    r_kwargs = dict(zip(parameters_names, args))
    r_kwargs.update(kwargs)

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
