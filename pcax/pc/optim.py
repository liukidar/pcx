__all__ = ["Optim"]

from ..core import Module as _Module
from ..core.parameters import Parameter, ParameterRef
import jax.numpy as jnp


class Optim(_Module):
    def __init__(self, optax_opt, vars, allow_none_grads=False):
        self.optax_opt = optax_opt
        self.train_refs = [ParameterRef(x) for x in vars]
        self.state = Parameter(
            self.optax_opt.init({id(var.ref): var.ref.value for var in self.train_refs})
        )
        self.allow_none_grads = allow_none_grads

    def __call__(self, grads):
        train_vars = {id(var.ref): var.ref.value for var in self.train_refs}
        if self.allow_none_grads:
            grads = {
                id(var.ref): grads.get(id(var.ref), None) for var in self.train_refs
            }
        else:
            grads = {id(var.ref): grads[id(var.ref)] for var in self.train_refs}
        updates, self.state.value = self.optax_opt.update(
            grads, self.state.value, train_vars
        )
        for var in self.train_refs:
            update = updates[id(var.ref)]
            if update is not None:
                var.ref.value += update

    @property
    def step_count(self) -> 'jnp.ndarray[int]':
        try:
            return self.state.value[-1][-1].count
        except AttributeError:
            return "Step count not available."
