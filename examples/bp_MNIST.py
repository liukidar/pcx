from pcax.core.node import NODE_STATUS, NODE_TYPE
import pcax.core as pxc
import pcax.nn as pxnn
import jax
import jax.numpy as jnp
import optax
import pcax.interface as pxi
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model(pxc.Module):
    linear1: pxnn.Linear
    linear2: pxnn.Linear
    linear3: pxnn.Linear
    linear4: pxnn.Linear

    def __init__(self, key, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        key, subkey = jax.random.split(key)
        self.linear1 = pxnn.Linear(input_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear2 = pxnn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear3 = pxnn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear4 = pxnn.Linear(hidden_dim, output_dim, _key=subkey)

    def init(self, state, x, t=None):
        return

    def __call__(self, x):
        act_fn = jax.nn.leaky_relu

        x = act_fn(self.linear1(x))
        x = act_fn(self.linear2(x))
        x = act_fn(self.linear3(x))
        x = self.linear4(x)

        y = x

        return y


def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


batch_size = 128
input_dim = 28 * 28
hidden_dim = 14 * 14 * 8
output_dim = 10

mnist_dataset = FashionMNIST(
    "/tmp/fashionmnist/", download=True, transform=FlattenAndCast()
)
training_generator = pxi.data.Dataloader(
    mnist_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
)

rseed = 0
rkey = jax.random.PRNGKey(rseed)
rkey, rsubkey = jax.random.split(rkey)


state = pxi.DefaultState()
trainer = pxi.Trainer()

state, model, optim = state.init(
    Model(rsubkey, input_dim, hidden_dim, output_dim),
    "*",
    batch_size=batch_size,
    input_shape=(input_dim,),
    optim_fn=lambda state: pxi.optim.combine(
        {
            NODE_TYPE.X: optax.sgd(0),
            NODE_TYPE.W: optax.chain(pxi.optim.reduce(), optax.adam(1e-4)),
        },
        state.get_masks("type"),
    ),
    trainer=trainer,
    init_fn_args=(None,),
)()


def loss_fn(state, model, y, t):
    return jnp.dot(y - t, y - t)


T = 1
E = 8


@pxi.jit
def run_on_batch(state, model, x, t, loss_fn, optim):
    model, _ = trainer.init_fn(state, model, x, t)

    (state, model), y = trainer.update_fn[NODE_TYPE.W](
        state=state,
        model=model,
        x_args=(x,),
        loss_fn_args=(t,),
        loss_fn=loss_fn,
        optim=optim,
    )

    target_class = jnp.argmax(t, axis=1)
    predicted_class = jnp.argmax(y[0], axis=1)
    accuracy = jnp.mean(predicted_class == target_class)

    return state, model, accuracy


batch_it = iter(training_generator)

epoch_times = []
accuracy_per_epoch = []
for e in range(E):
    accuracies = []
    start_time = time.time()
    for (x, y) in training_generator:
        state, model, accuracy = run_on_batch(loss_fn=loss_fn, optim=optim)(
            state, model, x, one_hot(y, output_dim)
        )
        accuracies.append(accuracy)

    epoch_time = time.time() - start_time
    if e > 1:
        epoch_times.append(epoch_time)

    accuracy_per_epoch.append(np.mean(accuracies))
    print("Epoch {} in {:0.3f} sec".format(e, epoch_time))
    print("Accuracy:", accuracy_per_epoch[-1])
print(f"Avg epoch time: {np.mean(epoch_times)}")

del training_generator
