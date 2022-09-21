import dataclasses
from typing import Dict, List, Any
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from pcax.core.energy import gaussian_energy

from pcax.utils.functions import ensure_list

from .node import NODE_TYPE, NodeModule
from .view import InputView, OutputView, View, TmpView


class Module(NodeModule):
    def __init__(self, **kwargs) -> None:
        # Initialize NodeModule with the given kwargs
        super().__init__(**kwargs)

    # Use 'view' when 'output_view' and 'input_view' are equal
    def __call__(
        self,
        mu: Any,
        output_view: str | List[str] = None,
        input_view: str | List[str] = None,
        view: str | List[str] = None,
    ) -> View:
        assert view is None or (
            output_view is None and input_view is None
        ), "You cannot set both 'view' and 'output/input_view' at the same time."

        if view is not None:
            input_view = output_view = view

        # mu is cached into the target output views
        output_views = self.at(output_view, type="output")
        output_views.set(mu)

        # TODO: call forward here?

        # return the input views
        return self.at(input_view, type="input")

    def energy(self, criterion) -> jnp.array:
        submodules = self.__get_submodules()

        # TODO: check more efficient versions (maybe jtu.reduce()?)
        e = list(map(lambda el: el.energy(criterion), submodules))
        energies = jnp.stack(e, axis=0)
        return jnp.sum(energies, axis=0)

    def at(self, view_name: str | List[str] = None, type=None) -> TmpView:
        r = []

        if view_name is None:
            for module in self.__get_submodules():
                r.extend(module.at(type=type))
        else:
            raise NotImplementedError()

        return r

    def _clear_cache(self):
        for submodule in self.__get_submodules():
            submodule._clear_cache()

    def _dummy_init(self, empty_x):
        for submodule in self.__get_submodules():
            submodule._dummy_init(empty_x)

    def __get_submodules(self):
        return (
            getattr(self, field.name)
            for field in dataclasses.fields(self)
            if issubclass(field.type, Module)
        )


class Layer(Module):
    x: jnp.ndarray
    view: View = eqx.static_field()
    default_views: Dict[str, List[str]] = eqx.static_field()

    def __init__(
        self,
        view: View = None,
        energy_view: str | List[str] = None,
        input_view: str | List[str] = None,
        output_view: str | List[str] = None,
    ):
        super().__init__(type=NODE_TYPE.X)

        self.x = jnp.empty(shape=(0,))  # Necessary so that pcax sees it as a parameter.
        if view is not None:
            self.view = view
        else:
            # Default view structure:
            self.view = View(
                _children=[
                    InputView(),
                    OutputView(
                        _energy_fn=lambda root, x, mu: gaussian_energy(
                            mu.get(root), x.get(root)
                        )
                    ),
                ]
            )

        self.default_views = {
            "energy": ensure_list(energy_view),
            "input": ensure_list(input_view),
            "output": ensure_list(output_view),
        }

    def energy(self, _criterion) -> jnp.array:
        energy_views = TmpView()
        # if self.default_views['energy'] is not None:
        # 	for ev in self.default_views['energy']:
        # 		energy_views.extend(self.view.match(ev, _type='view', _only_leaves=False))

        if len(energy_views) == 0:
            energy_views.append(self.view, self)

        # TODO: check more efficient versions
        return jtu.tree_reduce(
            lambda e1, e2: e1 + e2,
            list(map(lambda view: _criterion(view), energy_views)),
        )

    def at(self, views: str | List[str] = None, type=None) -> TmpView:
        r = TmpView()

        if views is None:
            if type is None:
                r.append(self.view, self)

                return r
            else:
                return self.at(self.default_views[type] or "*", type=type)
        elif isinstance(views, str):
            views = [views]

        for v in views:
            r.extend(map(lambda view: (self, view), self.view.match(v, type=type)))

        return r

    def _clear_cache(self):
        view_stack = [self.view]

        while len(view_stack) > 0:
            view = view_stack.pop()
            view.cached = None
            view_stack.extend(view.children)

    def _dummy_init(self, x):
        self.x.set(x)
