from pcax.core.energy import EnergyCriterion
from pcax.core.node import NODE_STATUS, NODE_TYPE
import pcax.core as pxc
import pcax.nn as pxnn
import jax
import jax.numpy as jnp
import optax
import pcax.interface as pxi
import numpy as np
from torchvision.datasets import FashionMNIST
import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# jax.config.update("jax_platform_name", "cpu")


class Model(pxc.Module):
    linear1: pxnn.Linear
    linear2: pxnn.Linear
    linear3: pxnn.Linear
    linear4: pxnn.Linear
    pc1: pxc.Layer
    pc2: pxc.Layer
    pc3: pxc.Layer
    pc4: pxc.Layer

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

        self.pc1 = pxc.Layer()
        self.pc2 = pxc.Layer()
        self.pc3 = pxc.Layer()
        self.pc4 = pxc.Layer()

    def init(self, x, t=None):
        with pxi.force_forward():
            return self(x, t)

    def __call__(self, x, t=None):
        act_fn = jax.nn.tanh

        x = self.pc1(act_fn(self.linear1(x)))
        x = self.pc2(act_fn(self.linear2(*x.get())))
        x = self.pc3(act_fn(self.linear3(*x.get())))
        x = self.pc4(act_fn(self.linear4(*x.get())))

        y = self.pc4.at(type="output").get()[0]

        # t should only be passed during initialization,
        # never during the forward pass.
        # the 'if' is used to avoid rewriting the code,
        # as initialization is done with a forward pass,
        # except for the last pc layer.
        if t is not None:
            # cache is disabled, we overwrite x with t
            self.pc4.at(type="output").set(t)

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

train_dataset = FashionMNIST(
    "/tmp/fashionmnist/",
    download=True,
    transform=FlattenAndCast(),
    train=True,
)
train_dataloader = pxi.data.Dataloader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
)

test_dataset = FashionMNIST(
    "/tmp/fashionmnist/",
    download=True,
    transform=FlattenAndCast(),
    train=False,
)
test_dataloader = pxi.data.Dataloader(
    train_dataset,
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
    batch_size=batch_size,
    input_shape=(input_dim,),
    optim_fn=lambda state: pxi.optim.combine(
        {
            NODE_TYPE.X: optax.chain(optax.sgd(0.05)),
            NODE_TYPE.W: optax.chain(pxi.optim.reduce(), optax.adam(1e-4)),
        },
        state.get_masks("type"),
    ),
    trainer=trainer,
    init_fn_args=(None,),
)()


def loss_fn(state, model, y, t):
    return model.energy(EnergyCriterion())


T = 3
E = 8
total_iterations = T * E


@pxi.jit
def run_on_batch(state, model, x, t, loss_fn, optim, mode="train"):
    model, y = trainer.init_fn(state, model, x, t)

    if mode == "train":
        state = state.update_mask("status", lambda mask: mask.pc4.x, NODE_STATUS.FROZEN)

        for j in range(T - 1):
            (state, model), _ = trainer.update_fn[NODE_TYPE.X](
                state,
                model,
                x_args=[x],
                loss_fn_args=[t],
                loss_fn=loss_fn,
                optim=optim,
            )

        (state, model), _ = trainer.update_fn[NODE_TYPE.W](
            state,
            model,
            x_args=[x],
            loss_fn_args=[t],
            loss_fn=loss_fn,
            optim=optim,
        )

    target_class = jnp.argmax(t, axis=1)
    predicted_class = jnp.argmax(y, axis=1)
    accuracy = jnp.mean(predicted_class == target_class)

    return state, model, accuracy


epoch_times = []
accuracy_per_epoch = []

# This slows down things but prints helpful error messages
# with pxi.debug():
for e in range(E):
    accuracies = []
    start_time = time.time()
    for (x, y) in train_dataloader:
        state, model, accuracy = run_on_batch(loss_fn=loss_fn, optim=optim, mode="test")(
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

del train_dataloader

# Check final accuracy on test set

accuracies = []
for (x, y) in test_dataloader:
    state, model, accuracy = run_on_batch(loss_fn=None, optim=None, mode="test")(
        state, model, x, one_hot(y, output_dim)
    )
    accuracies.append(accuracy)
print("Test accuracy:", np.mean(accuracies))

del test_dataloader
