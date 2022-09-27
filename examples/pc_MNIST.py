from pcax.core.energy import EnergyCriterion
from pcax.core.node import NODE_STATUS, NODE_TYPE
import pcax.core as pxc
import pcax.nn as pxnn
import jax
import jax.numpy as jnp
import optax
import pcax.interface as pxi
import numpy as np
from torchvision.datasets import MNIST
import time

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
jax.config.update("jax_platform_name", "cpu")


class Model(pxc.Module):
    linear1: pxnn.Linear
    linear2: pxnn.Linear
    linear3: pxnn.Linear
    pc1: pxc.Layer
    pc2: pxc.Layer
    pc3: pxc.Layer

    def __init__(self, key, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        key, subkey = jax.random.split(key)
        self.linear1 = pxnn.Linear(input_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear2 = pxnn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear3 = pxnn.Linear(hidden_dim, output_dim, _key=subkey)

        self.pc1 = pxc.Layer()
        self.pc2 = pxc.Layer()
        self.pc3 = pxc.Layer()

    def init(self, state, x, t=None):
        with pxi.force_forward():
            self(x, t)

    def __call__(self, x, t=None):
        act_fn = jax.nn.tanh

        x = self.pc1(act_fn(self.linear1(x)))
        x = self.pc2(act_fn(self.linear2(*x.get())))
        if t is None:
            x = self.pc3(act_fn(self.linear3(*x.get())))
            y = self.pc3.at(type="output").get()[0]
        else:
            y = self.pc3(t)

        return y


def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


batch_size = 256
input_dim = 28 * 28
hidden_dim = 256
output_dim = 10

mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
training_generator = pxi.data.Dataloader(mnist_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

rseed = 0
rkey = jax.random.PRNGKey(rseed)
rkey, rsubkey = jax.random.split(rkey)


state = pxi.DefaultState()
trainer = pxi.Trainer()

state, model, optim = state.init(
    Model(rsubkey, input_dim, hidden_dim, output_dim),
    batch_size=batch_size,
    input_shape=(input_dim,),
    optim_fn=lambda state: pxi.optim.combine(
        {
            NODE_TYPE.X: optax.sgd(0.05),
            NODE_TYPE.W: optax.chain(pxi.optim.reduce(), optax.adam(1e-3)),
        },
        state.get_masks("type"),
    ),
    trainer=trainer,
    init_fn_args=(None,),
)()


def loss_fn(state, model, y, t):
    return model.energy(EnergyCriterion())


T = 3
E = 16
total_iterations = T * E


@pxi.jit
def run_on_batch(state, model, x, t, loss_fn, optim):
    model = trainer.init_fn(state, model, x, t)
    state = state.update_mask("status", lambda mask: mask.pc3.x, NODE_STATUS.FROZEN)

    r, y = pxi.flow.scan(trainer.update_fn[NODE_TYPE.X], loss_fn=loss_fn, optim=optim, length=T - 1,)(
        state,
        model,
        x_args=(x,),
        loss_fn_args=(t,),
    )
    state, model = r["state"], r["model"]

    (state, model), _ = trainer.update_fn[NODE_TYPE.W](
        state,
        model,
        x_args=[x],
        loss_fn_args=[t],
        loss_fn=loss_fn,
        optim=optim,
    )

    target_class = jnp.argmax(t, axis=1)
    predicted_class = jnp.argmax(y[0][0, ...], axis=1)
    accuracy = jnp.mean(predicted_class == target_class)

    return state, model, accuracy


epoch_times = []
energies = []

# This slows down things but prints helpful error messages
with pxi.debug():
    for e in range(E):
        energy = []
        start_time = time.time()

        for (x, y) in training_generator:
            state, model, en = run_on_batch(loss_fn=loss_fn, optim=optim)(state, model, x, one_hot(y, output_dim))
            energy.append(en)

        epoch_time = time.time() - start_time
        if e > 1:
            epoch_times.append(epoch_time)

        energies.extend(np.array(energy).tolist())
        print("Epoch {} in {:0.3f} sec".format(e, epoch_time))
        print("Accuracy:", np.mean(energies))
    print(f"Avg epoch time: {np.mean(epoch_times)}")
