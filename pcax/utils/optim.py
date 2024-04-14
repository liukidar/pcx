__all__ = ["Optim"]

from jaxtyping import PyTree
import optax
import jax.tree_util as jtu
import equinox as eqx

from ..core._module import BaseModule
from ..core._parameter import Param, BaseParam, set, get
from ..core._static import static


class Optim(BaseModule):
    def __init__(
        self,
        optax_opt: optax.GradientTransformation,
        parameters: PyTree | None = None
    ):
        self.optax_opt = static(optax_opt)

        if parameters is not None:
            self.init(parameters)

    def step(self, module: PyTree, grads: PyTree) -> None:
        module = eqx.filter(module, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))
        grads = eqx.filter(grads, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))

        updates, state = self.optax_opt.update(
            grads,
            self.state.get(),
            module,
        )
        self.state.set(state)

        jtu.tree_map(
            lambda u, p: set(p, eqx.apply_updates(get(p), get(u))),
            updates,
            module,
            is_leaf=lambda x: isinstance(x, BaseParam)
        )

    def init(self, parameters: PyTree) -> None:
        self.filter = static(jtu.tree_map(
            lambda x: get(x) is not None,
            parameters,
            is_leaf=lambda x: isinstance(x, BaseParam)
        ))
        parameters = eqx.filter(parameters, self.filter.get(), is_leaf=lambda x: isinstance(x, BaseParam))

        self.state = Param(
            self.optax_opt.init(parameters)
        )
