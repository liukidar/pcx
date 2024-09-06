from typing import Callable
import math
from pathlib import Path
import logging
import sys
import argparse

# Core dependencies
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from pcax import RKG

sys.path.insert(0, "../../../")
from data_utils import get_vision_dataloaders, seed_everything, get_config_value  # noqa: E402

sys.path.pop(0)


def seed_pcax_and_everything(seed: int | None = None):
    if seed is None:
        seed = 0
    RKG.seed(seed)
    seed_everything(seed)


logging.basicConfig(level=logging.INFO)


STATUS_FORWARD = "forward"
ACTIVATION_FUNCS = {
    None: lambda x: x,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "hard_tanh": jax.nn.hard_tanh,
    "sigmoid": jax.nn.sigmoid,
}


class SkipError(pxc.EnergyModule):
    def __init__(
        self,
        num_layers: int,
        input_dim: tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        backward_indices: list,
        beta: float,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.act_fn = px.static(act_fn)
        self.backward_indices = px.static(backward_indices)
        self.beta = beta

        self.layer_dims = [math.prod(input_dim)] + [hidden_dim for _ in range(num_layers - 1)] + [num_classes]

        self.layers = []
        for layer_input, layer_output in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.layers.append(pxnn.Linear(layer_input, layer_output))

        self.backward_layers = [pxnn.Linear(num_classes, hidden_dim) for i in range(len(backward_indices))]

        self.vodes = []
        for layer_output in self.layer_dims[1:-1]:
            self.vodes.append(pxc.Vode())
        self.vodes.append(pxc.Vode(pxc.se_energy))
        self.vodes[-1].h.frozen = True

        # for index in self.backward_indices:
        #     self.vodes[index].h.frozen = True

    def __call__(self, x, y=None):
        x = x.flatten()

        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i < len(self.layers) - 1 else lambda x: x
            x = layer(x)
            x = act_fn(self.vodes[i](x))

        if y is not None:
            self.vodes[-1].set("h", y)

            error = self.vodes[-1].get("u") - y

            for i, index in enumerate(self.backward_indices):
                h_hat = self.backward_layers[i](error)
                self.vodes[index].set("h", self.vodes[index].get("h") - self.beta * h_hat)

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y=None, *, model: SkipError):
    return model(x, y)


@pxf.vmap(
    pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
    in_axes=(0,),
    out_axes=(None, 0),
    axis_name="batch",
)
def energy(x, *, model: SkipError):
    y_ = model(x)
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, x: jax.Array, y: jax.Array, *, model: SkipError, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0
):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to(([False, True])), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"])
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: SkipError):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: SkipError, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, jax.nn.one_hot(y, 10), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)


def eval(dl, *, model: SkipError):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def run_experiment(
    *,
    dataset_name: str = "cifar10",
    num_layers: int,
    hidden_dim: int,
    beta: float,
    num_classes: int = 10,
    act_fn: str | None,
    batch_size: int,
    epochs: int,
    T: int,
    optim_x_lr: float,
    optim_x_momentum: float,
    optim_w_name: str,
    optim_w_lr: float,
    optim_w_wd: float,
    optim_w_momentum: float,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> float:
    seed_pcax_and_everything(seed)

    # Channel first: (batch, channel, height, width)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    input_dim = dataset.train_dataset[0][0].shape

    model = SkipError(
        num_layers=num_layers,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        act_fn=ACTIVATION_FUNCS[act_fn],
        beta=beta,
        backward_indices=[2, 5],
    )

    # with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    #     forward(jnp.zeros((batch_size, math.prod(input_dim))), None, model=model)

    optim_h = pxu.Optim(optax.sgd(learning_rate=optim_x_lr, momentum=optim_x_momentum))
    mask = pxu.M(pxnn.LayerParam)(model)
    mask.backward_layers = jax.tree_util.tree_map(
        lambda x: None, mask.backward_layers, is_leaf=lambda x: isinstance(x, pxnn.LayerParam)
    )

    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd),
            mask,
        )
    elif optim_w_name == "sgd":
        optim_w = pxu.Optim(optax.sgd(learning_rate=optim_w_lr, momentum=optim_w_momentum), mask)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_w_name}")

    model_save_dir: Path | None = checkpoint_dir / dataset_name / "best_model" if checkpoint_dir is not None else None
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    print("Training...")

    best_acc: float | None = None
    test_acc: list[float] = []
    for epoch in range(epochs):
        train(
            dataset.train_dataloader,
            T=T,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )
        mean_acc, _ = eval(dataset.test_dataloader, model=model)
        if np.isnan(mean_acc):
            logging.warning("Model diverged. Stopping training.")
            break
        test_acc.append(mean_acc)
        if epochs > 1 and model_save_dir is not None and (best_acc is None or mean_acc >= best_acc):
            best_acc = mean_acc
        print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {mean_acc:.4f}")

    print(f"\nBest accuracy: {best_acc}")

    return min(test_acc) if test_acc else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/skiperror_cifar10_adamw_hypertune.yaml", help="Path to the config file."
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    run_experiment(
        dataset_name=get_config_value(config, "dataset_name"),
        seed=get_config_value(config, "seed", required=False),
        num_layers=get_config_value(config, "hp/num_layers"),
        hidden_dim=get_config_value(config, "hp/hidden_dim"),
        beta=get_config_value(config, "hp/beta"),
        act_fn=get_config_value(config, "hp/act_fn"),
        batch_size=get_config_value(config, "hp/batch_size"),
        epochs=get_config_value(config, "hp/epochs"),
        T=get_config_value(config, "hp/T"),
        optim_x_lr=get_config_value(config, "hp/optim/x/lr"),
        optim_x_momentum=get_config_value(config, "hp/optim/x/momentum"),
        optim_w_name=get_config_value(config, "hp/optim/w/name"),
        optim_w_lr=get_config_value(config, "hp/optim/w/lr"),
        optim_w_wd=get_config_value(config, "hp/optim/w/wd"),
        optim_w_momentum=get_config_value(config, "hp/optim/w/momentum"),
        checkpoint_dir=Path("results/skip_error"),
    )
