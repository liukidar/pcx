import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pcax.core.nn import NODE_STATUS, NODE_TYPE
from pcax.utils.inference import batch_over, compute_grad
import equinox as eqx

######
#
# pcax.Simple
#
######


def make_traceable(fn, fn_kwargs_dict={}, **fn_kwargs):
    return {
        key: lambda *args, **kwargs: fn(*args, **kwargs, **fn_kwargs, **value)
        for key, value in fn_kwargs_dict.items()
    }


class Trainer:
    @staticmethod
    @batch_over(
        batch_in_mask=(0,),
        batch_filter_fn=lambda node_info: (
            0 if node_info.type == NODE_TYPE.X else None
        ),
    )
    def init_fn(state, model, x):
        with state.unfreeze(
            model, lambda _, info: info.type == NODE_TYPE.X
        ) as unforzen_model:
            unforzen_model.init(state, x, jnp.ones_like(x).mean())

            model = state.freeze(unforzen_model)

        return state, model

    @staticmethod
    def update_fn(
        state,
        model,
        x_args=[],
        x_kwargs={},
        optim=None,
        grad_filter_fn=None,
        loss_fn=None,
        loss_fn_args=[],
        loss_fn_kwargs={},
    ):
        @compute_grad()
        @batch_over(
            batch_in_mask=(0, None, None, 0, None),
            batch_out_full_mask=(None, 0),
            batch_filter_fn=lambda node_info: (
                0 if node_info.type == NODE_TYPE.X else None
            ),
        )
        def forward(
            state, model, x_args, x_kwargs, loss_fn, loss_fn_args, loss_fn_kwargs
        ):
            x = model(*x_args, **x_kwargs)
            l = loss_fn(state, model, x, *loss_fn_args, **loss_fn_kwargs)

            return l, x

        (l, x), grad = forward(
            state,
            model,
            x_args,
            x_kwargs,
            loss_fn,
            loss_fn_args,
            loss_fn_kwargs,
            grad_filter_fn=grad_filter_fn,
        )

        updates, optim_state = optim.update([grad], state.masks["optim"], [model])
        state.masks["optim"] = optim_state

        return (state, eqx.apply_updates(model, updates[0]), x_args), l


######
#
# END: pcax.Simple
#
######

import pcax.core as pcax
import pcax.nn as nn
import jax.numpy as jnp
import optax
import jax
from pcax.utils.optim import multi_transform
from pcax.core.energy import EnergyCriterion
import numpy as np

from pcax.utils.state import create, init

# jax.config.update('jax_platform_name', 'cpu')


class Model(pcax.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear
    linear4: nn.Linear
    pc1: pcax.Layer
    pc2: pcax.Layer
    pc3: pcax.Layer
    pc4: pcax.Layer

    def __init__(self, key) -> None:
        super().__init__()

        input_dim = 4
        hidden_dim = 128
        output_dim = 1

        key, subkey = jax.random.split(key)
        self.linear1 = nn.Linear(input_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear4 = nn.Linear(hidden_dim, output_dim, _key=subkey)

        self.pc1 = pcax.Layer()
        self.pc2 = pcax.Layer()
        self.pc3 = pcax.Layer()
        self.pc4 = pcax.Layer()
        self.pc4._node_info.status = NODE_STATUS.FROZEN

    def init(self, state, input_data, output_data):
        self.pc1.x.set(self.linear1(input_data))
        self.pc2.x.set(self.linear2(self.pc1.x.get()))
        self.pc3.x.set(self.linear3(self.pc2.x.get()))
        self.pc4.x.set(output_data)

        return state

    def __call__(self, x):
        x = self.pc1(self.linear1(x))
        x = self.pc2(self.linear2(*x.get()))
        x = self.pc3(self.linear3(*x.get()))
        x = self.pc4(self.linear4(*x.get()))

        x = self.pc4.view.children[1].get(self.pc4)

        return x


def loss_fn(state, model, y):
    r = jax.lax.psum(model.energy(EnergyCriterion()), axis_name="__batch")

    return r


rseed = 0
rkey = jax.random.PRNGKey(rseed)
rkey, rsubkey = jax.random.split(rkey)

state, model = create(Model(rkey))
trainer = Trainer()

batch_size = 128
batches_per_epoch = 256
T = 32
epochs = 64
train_data = [jax.random.uniform(rsubkey, shape=(batch_size, 4))] * batches_per_epoch
state, model, optimizer = init(
    state,
    model,
    multi_transform(
        {NODE_TYPE.X: optax.sgd(1e-4), NODE_TYPE.W: optax.sgd(1e-4)},
        state.get_type_mask(),
    ),
    batch_size,
    (4,),
    trainer,
)


def run(state, model, optim, train_data):
    @jax.jit
    def step(state, model, data):
        state, model = trainer.init_fn(state, model, data)

        update = make_traceable(
            trainer.update_fn,
            {
                NODE_TYPE.X: {
                    "grad_filter_fn": (
                        lambda i: (
                            i.type == NODE_TYPE.X and i.status != NODE_STATUS.FROZEN
                        )
                    )
                },
                NODE_TYPE.W: {
                    "grad_filter_fn": (
                        lambda i: (
                            i.type == NODE_TYPE.W and i.status != NODE_STATUS.FROZEN
                        )
                    )
                },
            },
            loss_fn=loss_fn,
            optim=optim,
        )

        (state, model, _), loss = jax.lax.scan(
            lambda carry, x: jax.lax.cond(
                (x + 1) % 4, update[NODE_TYPE.X], update[NODE_TYPE.W], *carry
            ),
            (state, model, (data,)),
            np.arange(T),
        )

        return state, model, loss

    for i, batch in enumerate(train_data):
        print(f"\t batch{i}")
        state, model, loss = step(state, model, batch)

    return state, model, loss


for e in range(epochs):
    print(f"epoch {e}")
    state, model, loss = run(state, model, optimizer, train_data)
