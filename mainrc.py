from pcax.core.nn import NODE_TYPE
import pcaxrc.core as pcax
import pcaxrc.nn as nn
import jax
import jax.numpy as jnp
import optax
from pcaxrc.sli import DefaultState, Trainer
from pcaxrc.utils.optim import multi_transform, reduce

jax.config.update("jax_platform_name", "cpu")


class Model(pcax.Module):
    linear1: nn.Linear
    pc1: pcax.Layer

    def __init__(self, key) -> None:
        super().__init__()

        input_dim = 4
        hidden_dim = 128

        key, subkey = jax.random.split(key)
        self.linear1 = nn.Linear(input_dim, hidden_dim, _key=subkey)

        self.pc1 = pcax.Layer()

    def init(self, state, input_data):
        self.pc1.x.set(self.linear1(input_data))

        return state

    def __call__(self, x):
        x = self.pc1(self.linear1(x))
        x = self.pc1.view.children[1].get(self.pc1)

        return x


rseed = 0
rkey = jax.random.PRNGKey(rseed)
rkey, rsubkey = jax.random.split(rkey)

batch_size = 8
input_shape = (4,)

state = DefaultState()
trainer = Trainer()

state, model, optim = state.init(
    Model(rsubkey),
    "*",
    batch_size=batch_size,
    input_shape=input_shape,
    optim_fn=lambda state: multi_transform(
        {
            NODE_TYPE.X: optax.sgd(1e-4),
            NODE_TYPE.W: optax.chain(reduce, optax.sgd(1e-4)),
        },
        state.get_masks("type"),
    ),
    trainer=trainer,
)()

x = jnp.ones((batch_size,) + input_shape)
model = trainer.init_fn(state, model, x)
state, model, y, loss = trainer.update_fn(
    state,
    model,
    x_args=[x],
    loss_fn=lambda _, __, x: jax.lax.psum(jnp.sum(x), axis_name="AX_BATCH"),
    optim=optim,
    loss_fn_args=[],
    loss_fn_kwargs={},
    x_kwargs={},
)

pass
