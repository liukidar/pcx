__all__ = ["_"]


class _:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], _):
            self.args = args[0].args
        else:
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

    def __class_getitem__(cls, args):
        return _and(*args)

    def __add__(self, other):
        return _(self, other)

    def __sub__(self, other):
        return _and(self, _not(other))

    def __neg__(self):
        return _not(self)

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
        self.arg = arg

    def apply(self, var):
        if isinstance(self.arg, _):
            r = self.arg.apply(var)
        else:
            r = isinstance(var, self.arg)

        return not r


class _hasattr(_):
    def __init__(self, arg, **attrs):
        self.arg = arg
        self.attrs = attrs

    def apply(self, var):
        return self.arg.apply(var) and all(
            hasattr(var, attr) and getattr(var, attr) == value
            for attr, value in self.attrs.items()
        )
