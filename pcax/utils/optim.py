__all__ = ["Optim"]

from jaxtyping import PyTree
import optax
import jax.tree_util as jtu
import equinox as eqx

from ..core._module import BaseModule
from ..core._parameter import Param, BaseParam, Param
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
        updates, state = self.optax_opt.update(
            grads,
            self.state.get(),
            module
        )
        self.state.set(state)
        
        jtu.tree_map(
            lambda u, p: p.set(eqx.apply_updates(p.get(), u.get())),
            updates,
            module,
            is_leaf=lambda x: isinstance(x, BaseParam)
        )

    def init(self, parameters: PyTree) -> None:
        self.state = Param(
            self.optax_opt.init(parameters)
        )
