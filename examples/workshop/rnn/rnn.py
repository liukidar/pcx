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


jax.config.update("jax_debug_nans", True)

sys.path.insert(0, "../../")
from data_utils import seed_everything, get_config_value  # noqa: E402

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


def print_nans(msg: str, obj):
    if isinstance(obj, jax.Array):
        has_nans = jnp.isnan(obj).any()
    else:
        leafs, _ = jax.tree.flatten(obj)
        has_nans = jnp.any(jnp.array([jnp.isnan(x).any() for x in leafs]))
    jax.debug.print(msg + ": {x}", x=has_nans)


class PCRNN(pxc.EnergyModule):
    def __init__(
        self,
        max_seq_len: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()
        self.max_seq_len = px.static(max_seq_len)
        self.input_dim = px.static(input_dim)
        self.hidden_dim = px.static(hidden_dim)
        self.output_dim = px.static(output_dim)
        self.act_fn = px.static(act_fn)
        self.W_input = pxnn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hidden = pxnn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = pxnn.Linear(hidden_dim, output_dim, bias=False)
        self.hidden_vodes = [pxc.Vode() for _ in range(max_seq_len)]
        self.out_vodes = [pxc.Vode(pxc.se_energy) for _ in range(max_seq_len)]
        for v in self.out_vodes:
            v.h.frozen = True

    def __call__(self, x, y=None):
        seq_len = x.shape[0]
        input_dim = 1
        if len(x.shape) > 1:
            input_dim = x.shape[1]
        assert seq_len <= self.max_seq_len.get(), "Sequence length cannot exceed the number of VODEs."
        assert input_dim == self.input_dim.get(), "Input dimension mismatch."

        h = jnp.zeros(self.hidden_dim.get())
        preds = jnp.zeros((self.max_seq_len.get(), self.output_dim.get()))
        for t in range(seq_len):
            h = self.W_hidden(h) + self.W_input(x[t])
            h = self.hidden_vodes[t](h)
            # print_nans("h is NaN", h)
            h = self.act_fn(h)
            pred = self.W_out(h)
            self.out_vodes[t](pred)
            preds = preds.at[t].set(pred)
            # print_nans("pred is NaN", pred)

            if y is not None:
                self.out_vodes[t].set("h", y[t])

        # jax.debug.print("Preds: {p}", p=preds)
        return preds


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y=None, *, model: PCRNN):
    return model(x, y)


@pxf.vmap(
    pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
    in_axes=(0,),
    out_axes=(None, 0),
    axis_name="batch",
)
def energy(x, *, model: PCRNN):
    y_ = model(x)
    return jax.lax.psum(model.energy(), "batch"), y_


# @pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: PCRNN, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            e, g = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to(([False, True])), has_aux=True)(
                energy
            )(x, model=model)

        # print_nans("X update NaNs", g["model"])
        all_grads = jnp.array(jax.tree.flatten(g["model"])[0])
        jax.debug.print(
            "E: {e:.4f}; grad.mean: {gm:.4f}; grad.std: {gs:.4f}",
            e=e[0].mean(),
            gm=all_grads.mean(),
            gs=all_grads.std(),
        )
        optim_h.step(model, g["model"])
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)
    # print_nans("W update NaNs", g["model"])
    optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0])


# @pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: PCRNN):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model)

    jax.debug.print("Eval pred: {y}; exp: {e}", y=y_, e=y)

    return jnp.pow(y - y_[..., None], 2).mean()


def train(dl, T, *, model: PCRNN, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, *, model: PCRNN):
    mses = []
    for x, y in dl:
        mse = eval_on_batch(x, y, model=model)
        mses.append(mse)

    return np.mean(mses)


def get_dataloaders(
    max_seq_len: int = 100,
    train_size: int = 9000,
    test_size: int = 1000,
    batch_size: int = 2,
):

    class DataLoader:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __iter__(self):
            return zip(self.x, self.y)

    x = jnp.linspace(0, 8 * math.pi, train_size + test_size).reshape(-1, max_seq_len, 1)
    y = jnp.sin(x)
    split = train_size // max_seq_len
    train_x = jnp.array_split(x[:split], train_size / max_seq_len / batch_size)
    train_y = jnp.array_split(y[:split], train_size / max_seq_len / batch_size)
    test_x = jnp.array_split(x[split:], test_size / max_seq_len / batch_size)
    test_y = jnp.array_split(y[split:], test_size / max_seq_len / batch_size)
    return DataLoader(train_x, train_y), DataLoader(test_x, test_y)


def window_mask(array_size: int, window_size: int) -> jax.Array:
    assert array_size > window_size
    i = jnp.arange(array_size)
    lower_part = i[:, None] > i[None, :] - window_size // 2 - 1
    upper_part = i[:, None] < i[None, :] + window_size // 2 + window_size % 2
    mask = (lower_part & upper_part).astype(jnp.float32)
    return mask


def run_experiment(
    *,
    act_fn: str | None,
    max_seq_len: int,
    train_size: int,
    test_size: int,
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

    max_seq_len = 5

    # train_dl, test_dl = get_dataloaders(
    #     max_seq_len=max_seq_len,
    #     train_size=train_size,
    #     test_size=test_size,
    # )

    model = PCRNN(
        max_seq_len=max_seq_len,
        input_dim=1,
        hidden_dim=16,
        output_dim=1,
        act_fn=ACTIVATION_FUNCS[act_fn],
    )

    optim_h = pxu.Optim(optax.sgd(learning_rate=optim_x_lr, momentum=optim_x_momentum))
    mask = pxu.M(pxnn.LayerParam)(model)
    # mask.skip_error_layers = jax.tree_util.tree_map(
    #     lambda x: None, mask.skip_error_layers, is_leaf=lambda x: isinstance(x, pxnn.LayerParam)
    # )

    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd),
            mask,
        )
    elif optim_w_name == "sgd":
        optim_w = pxu.Optim(optax.sgd(learning_rate=optim_w_lr, momentum=optim_w_momentum), mask)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_w_name}")

    model_save_dir: Path | None = checkpoint_dir / "best_model" if checkpoint_dir is not None else None
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    print("Training...")

    x = jnp.arange(max_seq_len) % 2
    conv_kernel = jnp.array([3, 2, 1])
    y = jax.nn.standardize(jnp.convolve(x, conv_kernel))[None, :, None]
    x = x.reshape(1, -1, 1)

    print(y)

    best_mse: float | None = None
    test_mse: list[float] = []
    for epoch in range(epochs):
        train_on_batch(5, x, y, model=model, optim_w=optim_w, optim_h=optim_h)
        mse = eval_on_batch(x, y, model=model)
        print("MSE:", mse)

        continue
        train(
            train_dl,
            T=T,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )
        avg_mse = eval(test_dl, model=model)
        if np.isnan(avg_mse):
            logging.warning("Model diverged. Stopping training.")
            break
        test_mse.append(avg_mse)
        if epochs > 1 and model_save_dir is not None and (best_mse is None or avg_mse <= best_mse):
            best_mse = avg_mse
        print(f"Epoch {epoch + 1}/{epochs} - Test MSE: {avg_mse:.4f}")

    # print(f"\nBest MSE: {best_mse}")

    # return min(test_mse) if test_mse else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/rnn.yaml", help="Path to the config file.")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    run_experiment(
        act_fn=get_config_value(config, "hp/act_fn"),
        max_seq_len=get_config_value(config, "hp/max_seq_len"),
        train_size=get_config_value(config, "hp/train_size"),
        test_size=get_config_value(config, "hp/test_size"),
        epochs=get_config_value(config, "hp/epochs"),
        T=get_config_value(config, "hp/T"),
        optim_x_lr=get_config_value(config, "hp/optim/x/lr"),
        optim_x_momentum=get_config_value(config, "hp/optim/x/momentum"),
        optim_w_name=get_config_value(config, "hp/optim/w/name"),
        optim_w_lr=get_config_value(config, "hp/optim/w/lr"),
        optim_w_wd=get_config_value(config, "hp/optim/w/wd"),
        optim_w_momentum=get_config_value(config, "hp/optim/w/momentum"),
        checkpoint_dir=Path("results/rnn"),
    )
