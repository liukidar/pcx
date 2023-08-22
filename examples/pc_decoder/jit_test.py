import jax
import jax.numpy as jnp
import numpy as np
import optax

import pcax as px  # type: ignore
import pcax.core as pxc  # type: ignore
import pcax.nn as nn  # type: ignore
import pcax.utils as pxu  # type: ignore


class DummyModel(px.EnergyModule):
    def __init__(self):
        self.fc_nodes = [nn.Linear(2, 4), nn.Linear(4, 1)]
        self.pc_nodes = [px.Node(), px.Node()]
        self.pc_nodes[-1].x.frozen = True

    def __call__(self, input: jax.Array, target: jax.Array | None = None) -> jax.Array:
        if target is not None:
            self.pc_nodes[-1]["x"] = target
        x = self.fc_nodes[0](input)
        x = self.pc_nodes[0](x)["x"]
        x = self.fc_nodes[1](x)
        x = self.pc_nodes[1](x)["u"]
        return x


@pxu.vectorize(
    px.f(px.NodeParam, with_cache=True),
    in_axis=(0, 0),
    out_axis=("sum",),
)  # type: ignore
def model_energy_loss(
    input: jax.Array, target: jax.Array, *, model: DummyModel
) -> jax.Array:
    model(input=input, target=target)
    res = model.energy()
    assert isinstance(res, jax.Array)
    return res


def build_train():
    grad_and_values = pxu.grad_and_values(
        px.f(px.NodeParam)(frozen=False) | px.f(px.LayerParam),  # type: ignore
    )(model_energy_loss)

    def train_model(
        inputs: jax.Array,
        targets: jax.Array,
        *,
        model: DummyModel,
        optimizer: pxu.Optim
    ) -> jax.Array:
        with pxu.step(model):
            gradients, _ = grad_and_values(inputs, targets, model=model)
            optimizer(gradients)
            return model.energy()

    return train_model


@pxu.jit()
def train_on_batch(
    inputs: jax.Array, targets: jax.Array, *, model: DummyModel, optimizer: pxu.Optim
) -> jax.Array:
    build_train()(inputs, targets, model=model, optimizer=optimizer)


def train():
    model = DummyModel()
    batch_size = 10
    with pxu.train(model, jnp.zeros((batch_size, 2))):
        optimizer = pxu.Optim(
            optax.sgd(1e-3),
            model.parameters(),
            allow_none_grads=True,
        )
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    inputs = jax.random.normal(subkey1, (batch_size, 2))
    targets = jax.random.normal(subkey2, (batch_size, 1))
    for _ in range(5):
        train_on_batch(inputs, targets, model=model, optimizer=optimizer)
    print("Done.")


if __name__ == "__main__":
    train()
