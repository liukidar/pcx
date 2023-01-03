__all__ = ["_"]


class _:
    def __init__(self, *args):
        self.args = args

    def apply(self, var):
        for f in self.args:
            if isinstance(f, _):
                r = f.apply(var)
            else:
                r = isinstance(var, f)

            if r is True:
                return True

        return False

    def __or__(self, other):
        return _(self, other)

    def __and__(self, other):
        return _and(self, other)

    def __invert__(self):
        return _not(self)

    def __sub__(self, other):
        return _and(self, ~other)

    def __call__(self, **kwargs):
        return _hasattr(self, **kwargs)


class _and(_):
    def __init__(self, *args):
        self.args = args

    def apply(self, var):
        for f in self.args:
            if isinstance(f, _):
                r = f.apply(var)
            else:
                r = isinstance(var, f)

            if r is False:
                return False

        return True


class _not(_):
    def __init__(self, arg):
        self.args = arg

    def apply(self, var):
        if isinstance(self.args, _):
            r = self.args.apply(var)
        else:
            r = isinstance(var, self.args)

        return not r


class _hasattr(_):
    def __init__(self, arg, **attrs):
        self.args = arg
        self.attrs = attrs

    def apply(self, var):
        return self.args.apply(var) and all(
            hasattr(var, attr) and getattr(var, attr) == value
            for attr, value in self.attrs.items()
        )
