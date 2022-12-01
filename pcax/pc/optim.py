__all__ = ["Optim"]

from ..core import Module as _Module, ModuleList, TrainRef, StateVar
import jax.tree_util as jt


def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _is_none(x):
    return x is None


def combine(*pytrees):
    return jt.tree_map(_combine, *pytrees, is_leaf=_is_none)


class Optim(_Module):
    def __init__(self, optax_opt, vars, allow_none_grads=False):
        self.optax_opt = optax_opt
        self.train_refs = ModuleList(TrainRef(x) for x in vars)
        self.state = StateVar(
            self.optax_opt.init({id(var.ref): var.ref.value for var in self.train_refs})
        )

        self.allow_none_grads = allow_none_grads

        if self.allow_none_grads:
            raise NotImplementedError()

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
