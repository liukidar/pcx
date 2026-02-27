import io
from contextlib import redirect_stderr, redirect_stdout

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import torch
import torchvision
import torchvision.transforms as transforms

import pcx as px
import pcx.functional as pxf
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu


class GroupNorm(px.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = px.static(num_groups)
        self.eps = px.static(eps)
        self.weight = pxnn.LayerParam(jnp.ones(num_channels))
        self.bias = pxnn.LayerParam(jnp.zeros(num_channels))

    def __call__(self, x: jax.Array) -> jax.Array:
        c = x.shape[0]
        g = self.num_groups.get()
        xg = x.reshape(g, c // g, *x.shape[1:])
        axes = tuple(range(1, xg.ndim))
        mean = jnp.mean(xg, axis=axes, keepdims=True)
        var = jnp.var(xg, axis=axes, keepdims=True)
        xg = (xg - mean) / jnp.sqrt(var + self.eps.get())
        xg = xg.reshape(x.shape)
        shape = (c,) + (1,) * (x.ndim - 1)
        return xg * self.weight.get().reshape(shape) + self.bias.get().reshape(shape)


class ResNetLayer(px.Module):
    def __init__(self, n_ch: int, n_inner: int, ks: int = 3, ng: int = 8, init_scale: float = 0.01):
        super().__init__()
        p = ks // 2
        self.conv1 = pxnn.Conv2d(n_ch, n_inner, ks, padding=(p, p), use_bias=False)
        self.conv2 = pxnn.Conv2d(n_inner, n_ch, ks, padding=(p, p), use_bias=False)
        self.norm1 = GroupNorm(ng, n_inner)
        self.norm2 = GroupNorm(ng, n_ch)
        self.norm3 = GroupNorm(ng, n_ch)
        self.init_scale = px.static(init_scale)

        for mod in (self.conv1, self.conv2):
            leaves = jtu.tree_leaves(mod, is_leaf=lambda x: isinstance(x, pxnn.LayerParam))
            for leaf in leaves:
                if isinstance(leaf, pxnn.LayerParam):
                    leaf.set(jax.random.normal(px.RKG(), leaf.shape) * self.init_scale.get())

    def __call__(self, z: jax.Array, x: jax.Array) -> jax.Array:
        y = self.norm1(jax.nn.relu(self.conv1(z)))
        return self.norm3(jax.nn.relu(z + self.norm2(x + self.conv2(y))))


def nudged_ce_energy(nudging: float):
    def energy_fn(vode, rkg=px.RKG):
        return nudging * (-(vode.get("h") * jax.nn.log_softmax(vode.get("u"))))

    return energy_fn


class DEQPCModel(pxc.EnergyModule):
    def __init__(self, n_channels: int = 48, n_inner: int = 64, n_classes: int = 10, nudging: float = 0.01, init_scale: float = 0.01):
        super().__init__()
        self.n_classes = px.static(n_classes)
        self.n_channels = px.static(n_channels)
        self.nudging = px.static(nudging)
        self.init_scale = px.static(init_scale)

        self.input_conv = pxnn.Conv2d(3, n_channels, 3, padding=(1, 1), use_bias=True)
        self.gn_in = GroupNorm(8, n_channels)
        self.f = ResNetLayer(n_channels, n_inner, init_scale=init_scale)
        self.gn_out = GroupNorm(8, n_channels)
        self.pool = pxnn.AvgPool2d(kernel_size=8, stride=8)
        self.linear = pxnn.Linear(n_channels * 4 * 4, n_classes)

        self.vode_z = pxc.Vode()
        self.vode_out = pxc.Vode(nudged_ce_energy(nudging))
        self.vode_out.h.frozen = True

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


_w_mask = pxu.M(pxnn.LayerParam).to([False, True])


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: DEQPCModel, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda x_i: model.gn_in(model.input_conv(x_i)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y)

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(T, model.vode_z.h.get(), model.x_inj_cache.get(), model.vode_out.h.get(), h_opt, model)
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, grads = pxf.value_and_grad(_w_mask, has_aux=False)(energy_weight)(x, model=model)

    optim_w.step(model, grads["model"], scale_by=1.0 / x.shape[0])


@pxf.jit(static_argnums=0)
def eval_on_batch(T: int, x: jax.Array, y_oh: jax.Array, *, model: DEQPCModel, optim_h: pxu.Optim):
    model.eval()

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x_inj = jax.vmap(lambda x_i: model.gn_in(model.input_conv(x_i)))(x)
        model.x_inj_cache.set(x_inj)
        model.vode_z.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(jnp.zeros_like(y_oh))

    h_opt = optim_h.optax_opt_fn()
    z_h = _run_inference_loop(
        T,
        model.vode_z.h.get(),
        model.x_inj_cache.get(),
        jnp.zeros_like(y_oh),
        h_opt,
        model,
    )
    model.vode_z.h.set(z_h)

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        y_pred = predict_cached(model=model).argmax(axis=-1)

    return (y_pred == y_oh.argmax(axis=-1)).mean()


def _quiet_cifar10(*, root, train, transform, download):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )


def get_dataloaders(train_batch_size: int, test_batch_size: int, root: str = "~/tmp/cifar10/"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = _quiet_cifar10(root=root, train=True, transform=transform, download=True)
    test_ds = _quiet_cifar10(root=root, train=False, transform=transform, download=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True
    )
    return train_dl, test_dl
