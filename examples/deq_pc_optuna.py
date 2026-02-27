"""
DEQ-PC: Deep Equilibrium via Predictive Coding on CIFAR-10.
(Optimised with jax.lax.fori_loop for the inference inner loop)

Optuna hyperparameter search version.
"""

import io
from contextlib import redirect_stderr, redirect_stdout
import time

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import torch
import torchvision
import torchvision.transforms as transforms

import pcx as px
import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

import optuna


px.RKG.seed(0)


# ── GroupNorm ────────────────────────────────────────────────────────────────


class GroupNorm(px.Module):
    """Channel-first group normalisation with learnable affine."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = px.static(num_groups)
        self.eps = px.static(eps)
        self.weight = pxnn.LayerParam(jnp.ones(num_channels))
        self.bias = pxnn.LayerParam(jnp.zeros(num_channels))

    def __call__(self, x: jax.Array) -> jax.Array:
        C = x.shape[0]
        G = self.num_groups.get()
        g = x.reshape(G, C // G, *x.shape[1:])
        axes = tuple(range(1, g.ndim))
        mean = jnp.mean(g, axis=axes, keepdims=True)
        var = jnp.var(g, axis=axes, keepdims=True)
        g = (g - mean) / jnp.sqrt(var + self.eps.get())
        g = g.reshape(x.shape)
        s = (C,) + (1,) * (x.ndim - 1)
        return g * self.weight.get().reshape(s) + self.bias.get().reshape(s)


# ── ResNet cell: f(z, x) ────────────────────────────────────────────────────


class ResNetLayer(px.Module):
    """f(z, x) = norm3(relu(z + norm2(x + conv2(norm1(relu(conv1(z)))))))"""

    def __init__(self, n_ch: int, n_inner: int, ks: int = 3, ng: int = 8):
        super().__init__()
        p = ks // 2
        self.conv1 = pxnn.Conv2d(n_ch, n_inner, ks, padding=(p, p), use_bias=False)
        self.conv2 = pxnn.Conv2d(n_inner, n_ch, ks, padding=(p, p), use_bias=False)
        self.norm1 = GroupNorm(ng, n_inner)
        self.norm2 = GroupNorm(ng, n_ch)
        self.norm3 = GroupNorm(ng, n_ch)
        for mod in (self.conv1, self.conv2):
            for leaf in jtu.tree_leaves(
                mod, is_leaf=lambda x: isinstance(x, pxnn.LayerParam)
            ):
                if isinstance(leaf, pxnn.LayerParam):
                    leaf.set(jax.random.normal(px.RKG(), leaf.shape) * 0.01)

    def __call__(self, z: jax.Array, x: jax.Array) -> jax.Array:
        y = self.norm1(jax.nn.relu(self.conv1(z)))
        return self.norm3(jax.nn.relu(z + self.norm2(x + self.conv2(y))))


# ── Nudged CE energy ─────────────────────────────────────────────────────────


def nudged_ce_energy(nudging: float):
    def energy_fn(vode, rkg=px.RKG):
        return nudging * (-(vode.get("h") * jax.nn.log_softmax(vode.get("u"))))

    return energy_fn


# ── DEQ-PC Model ─────────────────────────────────────────────────────────────


class DEQPCModel(pxc.EnergyModule):
    def __init__(
        self,
        n_channels: int = 48,
        n_inner: int = 64,
        n_classes: int = 10,
        nudging: float = 0.1,
    ):
        super().__init__()
        self.n_classes = px.static(n_classes)
        self.n_channels = px.static(n_channels)
        self.nudging = px.static(nudging)

        # Embedding layers
        self.input_conv = pxnn.Conv2d(3, n_channels, 3, padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)

        # Weight-tied residual cell
        self.f = ResNetLayer(n_channels, n_inner)

        # Classification head
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=8, stride=8)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        # PC nodes
        self.vode_z = pxc.Vode()
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

        # Cached embedding — frozen so it is excluded from h/w gradient masks.
        self.x_inj_cache = pxc.VodeParam()
        self.x_inj_cache.frozen = True

    def embed(self, x: jax.Array) -> jax.Array:
        x_inj = self.gn_in(self.input_conv(x))
        self.x_inj_cache.set(x_inj)
        return x_inj

    def __call__(self, y: jax.Array | None = None) -> jax.Array:
        x_inj = self.x_inj_cache.get()

        z = self.vode_z.get("h")
        self.vode_z(self.f(z, x_inj))

        z_out = self.pool(self.gn_out(z)).flatten()
        logits = self.linear(z_out)
        self.vode_out(logits)

        if y is not None:
            self.vode_out.set("h", y)

        return logits

    def forward_full(self, x: jax.Array, y: jax.Array | None = None) -> jax.Array:
        self.embed(x)
        return self(y)


# ── PCX functional wrappers ──────────────────────────────────────────────────

_V = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))


@pxf.vmap(_V, in_axes=(0, 0), out_axes=0)
def init_train(x, y, *, model: DEQPCModel):
    x_inj = model.embed(x)
    model.vode_z.h.set(jnp.zeros_like(x_inj))
    return model(y)


@pxf.vmap(_V, in_axes=(0,), out_axes=0)
def init_eval(x, *, model: DEQPCModel):
    x_inj = model.embed(x)
    model.vode_z.h.set(jnp.zeros_like(x_inj))
    model.vode_out.h.set(jnp.zeros(model.n_classes.get()))
    return model()


@pxf.vmap(_V, in_axes=(0,), out_axes=None, axis_name="batch")
def energy_weight(x, *, model: DEQPCModel):
    model.forward_full(x)
    return jax.lax.psum(model.energy(), "batch")


@pxf.vmap(_V, in_axes=(), out_axes=0)
def predict_cached(*, model: DEQPCModel):
    return model()


# ── fori_loop inference ──────────────────────────────────────────────────────


def _run_inference_loop(T, z_h, x_inj, label_h, h_opt, model):
    h_opt_state = h_opt.init(z_h)
    nudging = model.nudging.get()

    @jax.checkpoint
    def per_sample_energy(z, x_i, lab):
        u_z = model.f(z, x_i)
        diff = z - u_z
        e_z = (0.5 * diff * diff).sum()

        z_out = model.pool(model.gn_out(z)).flatten()
        logits = model.linear(z_out)
        e_out = (nudging * (-(lab * jax.nn.log_softmax(logits)))).sum()

        return e_z + e_out

    per_sample_grad = jax.vmap(jax.grad(per_sample_energy))

    def body(_, carry):
        z_h, opt_state = carry
        grad_z = per_sample_grad(z_h, x_inj, label_h)
        updates, opt_state = h_opt.update(grad_z, opt_state, z_h)
        z_h = optax.apply_updates(z_h, updates)
        return z_h, opt_state

    z_h, _ = jax.lax.fori_loop(0, T, body, (z_h, h_opt_state))
    return z_h


# ── Gradient mask for weight update ──────────────────────────────────────────

_w_mask = pxu.M(pxnn.LayerParam).to([False, True])


# ── Train / eval steps ──────────────────────────────────────────────────────


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: DEQPCModel,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim,
):
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        init_train(x, y, model=model)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T,
        model.vode_z.h.get(),
        model.x_inj_cache.get(),
        model.vode_out.h.get(),
        h_opt,
        model,
    )
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(_w_mask, has_aux=False)(energy_weight)(x, model=model)
    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch(
    T: int, x: jax.Array, y: jax.Array, *, model: DEQPCModel, optim_h: pxu.Optim,
):
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        init_eval(x, model=model)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T,
        model.vode_z.h.get(),
        model.x_inj_cache.get(),
        model.vode_out.h.get(),
        h_opt,
        model,
    )
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_ = predict_cached(model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


# ── Data ─────────────────────────────────────────────────────────────────────


def _quiet_cifar10(*, root, train, transform, download):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download,
        )


def get_dataloaders(batch_size: int):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = _quiet_cifar10(root="~/tmp/cifar10/", train=True, transform=train_transform, download=True)
    test_ds = _quiet_cifar10(root="~/tmp/cifar10/", train=False, transform=test_transform, download=True)

    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True),
    )


# ── Epoch helpers ────────────────────────────────────────────────────────────


def train_epoch(dl, T, *, model, optim_w, optim_h):
    for x, y in dl:
        train_on_batch(
            T,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), model.n_classes.get()),
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )


def evaluate(dl, T, *, model, optim_h):
    accs = []
    for x, y in dl:
        a, _ = eval_on_batch(
            T, x.numpy(), y.numpy(), model=model, optim_h=optim_h,
        )
        accs.append(a)
    return float(np.mean(accs))


# ── Optuna search ────────────────────────────────────────────────────────────


def make_objective(train_dl, test_dl, *, n_epochs: int, batch_size: int, chan: int = 48):

    # Baseline (“averages”) taken from your file
    baseline = dict(
        T_train=150,
        nudging=0.05,
        lr_w=0.0025,
        wd_w=0.005,
        lr_h=0.25,
        mom_h=0.5,
    )

    def objective(trial: optuna.Trial) -> float:
        # Re-seed per trial for comparability (optional but usually helpful)
        px.RKG.seed(0)

        # ---- Search space (exactly as you requested) ----
        T_train = trial.suggest_int("T_train", 100, 300)
        nudging = trial.suggest_float("nudging", 0.001, 0.05, log=True)

        lr_w = trial.suggest_float("lr_w", 0.0005, 0.005, log=True)
        wd_w = trial.suggest_float("wd_w", 0.0001, 0.1, log=True)

        lr_h = trial.suggest_float("lr_h", 0.1, 1.0, log=True)
        mom_h = trial.suggest_float("mom_h", 0.0, 0.9)

        # If you later want a separate eval T, you can re-enable this:
        # T_eval = trial.suggest_int("T_eval", 20, 80)
        # For now, use training T for evaluation for consistency:
        T_eval = T_train

        # ---- Build model/opts for this trial ----
        model = DEQPCModel(n_channels=chan, n_inner=64, n_classes=10, nudging=nudging)

        optim_h = pxu.Optim(lambda: optax.sgd(lr_h, momentum=mom_h, nesterov=True))

        steps_per_epoch = len(train_dl)
        schedule = optax.piecewise_constant_schedule(
            init_value=lr_w,
            boundaries_and_scales={
                20 * steps_per_epoch: 0.2,
                40 * steps_per_epoch: 0.2,
            }
        )
        optim_w = pxu.Optim(
            lambda: optax.adamw(schedule, weight_decay=wd_w),
            pxu.M(pxnn.LayerParam)(model),
        )

        # Warmup: establish VodeParam shapes for vmap splitting.
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            init_train(
                jnp.zeros((batch_size, 3, 32, 32)),
                jnp.zeros((batch_size, 10)),
                model=model,
            )

        best = 0.0
        for epoch in range(1, n_epochs + 1):
            train_epoch(train_dl, T_train, model=model, optim_w=optim_w, optim_h=optim_h)
            acc = evaluate(test_dl, T_eval, model=model, optim_h=optim_h)

            best = max(best, acc)
            trial.report(acc, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best

    return objective, baseline


def main():
    # Fixed for the search (as you requested)
    batch_size = 256
    n_epochs = 20

    # Data once (avoid re-downloading / re-building workers each trial)
    train_dl, test_dl = get_dataloaders(batch_size)

    objective, baseline = make_objective(train_dl, test_dl, n_epochs=n_epochs, batch_size=batch_size, chan=48)

    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    storage = "sqlite:///deqpc_optuna_new.db"
    study_name = "deqpc_cifar10"

    # If the DB + study_name already exist, this resumes instead of starting over.
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    # Only enqueue the baseline once (otherwise it will be added every rerun).
    if len(study.trials) == 0:
        study.enqueue_trial(baseline)

    # Use the file’s hyperparams as the “average” point by enqueueing it.
    study.enqueue_trial(baseline)

    # Run search
    study.optimize(objective, n_trials=150, gc_after_trial=True, show_progress_bar=True)

    print("\nBest value (accuracy):", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()