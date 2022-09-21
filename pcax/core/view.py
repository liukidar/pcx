from typing import Callable, List, Any, Type, Tuple, Union
import jax.numpy as jnp

from pcax.lib.state import Param


class View:
    name: str
    parent: Union["View", jnp.ndarray]
    children: List["View"]
    transformation_fn: Callable
    cached: jnp.ndarray
    # boundaries: Any = eqx.static_field() # TODO

    def __init__(
        self,
        _name: str = None,
        _transformation_fn: Callable = None,
        _children: List["View"] = None,
        _boundaries=None,
    ) -> None:
        self.name = _name
        self.parent = None
        self.children = _children if _children is not None else []
        self.transformation_fn = _transformation_fn
        self.cached = None

        if _boundaries is not None:
            raise NotImplementedError(
                "Boundaries for a View are not currently supported."
            )

        for child in self.children:
            child.parent = self

    def get(
        self, root: jnp.ndarray, _apply_t: bool = False, _cache: bool = False
    ) -> jnp.ndarray:
        if _cache is True and self.cached is not None:
            return self.cached

        if isinstance(self.parent, View):
            x = self.parent.get(root)
        else:
            # Convert root.x to a Param if it is not already to get an uniform interface
            x = Param(root.x).get()

        if _apply_t is True and self.transformation_fn is not None:
            x = self.transformation_fn(x)

        if _cache is True:
            self.cached = x

        return x

    def set(self, root: jnp.ndarray, _x: jnp.ndarray, _cache: bool = False):
        if isinstance(self.parent, View):
            self.parent.set(root, _x, _cache)
        else:
            # x can be set only if it is a Param
            root.x.set(_x)

        if self.transformation_fn is not None:
            _x = self.transformation_fn(_x)
        if _cache is True:
            self.cached = _x

        return self.parent

    def flush_cache(self):
        self.cached = None

        for child in self.children:
            child.flush_cache()

    def match(
        self, name: str, type: str = None, _only_leaves: bool = True
    ) -> "TmpView":
        name2type = {"view": View, "input": InputView, "output": OutputView}
        r = []

        name = name.split("/", 1)
        next_name = name[0]
        if name[0] == self.name or name[0] == "*":
            if len(name) > 1:
                next_name = name[1]
            elif (
                _only_leaves is False
                or len(self.children) == 0
                and (type is None or isinstance(self, name2type[type]))
            ):
                r.append(self)

        for child in self.children:
            r.extend(child.match(next_name, type, _only_leaves))

        return r

    def find(self, _path: str) -> "View":
        _path = _path.split("/", 1)
        target = None

        if _path[0] == ".":
            target = self
        elif _path[0] == "..":
            assert isinstance(self.parent, View)
            target = self.parent
        else:
            for child in self.children:
                if child.name == _path[0]:
                    target = child
                    break

        if len(_path) == 1:
            return target
        else:
            return target.find(_path[1])

    def clone(
        self,
        _target: str = None,
        _name: str = None,
        _type: Type["View"] = None,
        **kwargs
    ):
        if _target is None:
            _target = self
        elif _target == "..":
            assert isinstance(self.parent, View)
            _target = self.parent

            if _name is None:
                raise NotImplementedError("A valid name needs to be specified")
        elif _target == ".":
            assert isinstance(self.parent, View)
            self.parent.children.remove(self)
            _target = self.parent

        if _type is None:
            _type = type(self)

        if _name is None:
            _name = self._name

        new_view = _type(_name, **kwargs)
        new_view.parent = self

        _target.append(new_view)

    def split(
        indices: Any,
        target: str = None,
        type: Type["View"] | Tuple[Type["View"]] = None,
        name: str | Tuple[str] = None,
        *args,
        **kwargs
    ):
        raise NotImplementedError()


class OutputView(View):
    energy_fn: None | Callable

    def __init__(
        self,
        _name: str = None,
        _transformation_fn: Callable = None,
        _children: Tuple["View"] = None,
        _boundaries=None,
        _energy_fn: Callable = None,
    ) -> None:
        super().__init__(_name, _transformation_fn, _children, _boundaries)
        self.energy_fn = _energy_fn

    def set(self, root: jnp.ndarray, _x: jnp.ndarray, _cache: bool = None):
        self.cached = (
            self.transformation_fn(_x) if self.transformation_fn is not None else _x
        )

    def get(
        self, root: jnp.ndarray, _cat_mode: str = "cat", _cache: bool = True, **_kwargs
    ) -> jnp.ndarray | Tuple[jnp.ndarray]:
        if _cache is True and self.cached is not None:
            return self.cached

        r = list((child.get(_cat_mode, _cache, **_kwargs) for child in self.children))

        if _cat_mode == "cat":
            args = {"axis": _kwargs.get("axis", -1)}
            r = jnp.concatenate(r, **args)

        if _cache is True:
            self.cached = r

        return r

    def match(
        self, _name: str, type: str = None, _only_leaves: bool = True
    ) -> "TmpView":
        if type is not None and type != "output":
            return TmpView()
        else:
            return super().match(_name, type, _only_leaves)

    def clone(self, _target: str = None, _name: str = None, **_kwargs):
        return super(View, self).clone(_target, _name, type=OutputView, **_kwargs)


class InputView(View):
    def get(self, root: jnp.ndarray, _cache: bool = True) -> jnp.ndarray:
        return super().get(root, _apply_t=True, _cache=_cache)

    def match(
        self, _name: str, type: str = None, _only_leaves: bool = True
    ) -> "TmpView":
        if type is not None and type != "input":
            return TmpView()
        else:
            return super().match(_name, type, _only_leaves)

    def clone(self, _target: str = None, _name: str = None, **_kwargs):
        return super(View, self).clone(_target, _name, type=InputView, **_kwargs)


class TmpView(View):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def append(self, view: View, root):
        self.children.append((root, view))

    def extend(self, views):
        self.children.extend(views)

    def get(self, *args, **kwargs) -> jnp.ndarray:
        return tuple(
            map(
                lambda root_view: root_view[1].get(root_view[0], *args, **kwargs),
                self.children,
            )
        )

    def set(self, x: jnp.ndarray, *args, **kwargs):
        if isinstance(x, tuple) and len(x) == len(self.children):
            xs = x
        else:
            xs = (x,) * len(self.children)

        for (child, value) in zip(self.children, xs):
            root, view = child
            view.set(root, value, *args, **kwargs)

        return

    def flush_cache(self):
        raise NotImplementedError("Private method")

    def clone(
        self,
        _target: str = None,
        _name: str = None,
        _type: Type["View"] = None,
        **kwargs
    ):
        raise NotImplementedError("Private method")
