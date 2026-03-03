import time
import os
import sys

import jax
import jax.numpy as jnp
import optax

import pcx as px
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

from deq.deq_pc_eval_minimal import evaluate_epoch
from deq.deq_pc_minimal_core import DEQPCModel, get_dataloaders, train_on_batch


# Tunable parameters
SEED = 0
N_EPOCHS = 15
T_STEPS = 100
N_CLASSES = 10
N_CHANNELS = 48
N_INNER = 64
NUDGING = 0.01
INIT_SCALE = 0.001

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
TRAIN_EVAL_SAMPLES = 10_000
DATA_ROOT = "~/tmp/cifar10/"

LR_W = 5e-4
WEIGHT_DECAY_W = 5e-3
LR_H = 0.25
MOMENTUM_H = 0.5
H_LR_DECAY = 0.97
STOP_GRAD_F = True


def assert_gpu_backend():
    print(f"Python executable: {sys.executable}")
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend} | devices: {devices}")
    if backend != "gpu":
        raise RuntimeError(
            "GPU backend required, but JAX is running on CPU. "
            "Install a CUDA-enabled jax/jaxlib build in this environment."
        )


def train_epoch(train_dl, *, model, optim_w, optim_h):
    for x, y in train_dl:
        train_on_batch(
            T_STEPS,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )


def main():
    px.RKG.seed(SEED)
    assert_gpu_backend()

    train_dl, test_dl = get_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, root=DATA_ROOT)

    model = DEQPCModel(
        n_channels=N_CHANNELS,
        n_inner=N_INNER,
        n_classes=N_CLASSES,
        nudging=NUDGING,
        init_scale=INIT_SCALE,
        stop_grad_f=STOP_GRAD_F,
    )

    schedule_h = optax.exponential_decay(
        init_value=LR_H,
        transition_steps=1,
        decay_rate=H_LR_DECAY,
    )
    optim_h = pxu.Optim(lambda: optax.sgd(schedule_h, momentum=MOMENTUM_H, nesterov=True))

    steps_per_epoch = len(train_dl)
    schedule_w = optax.piecewise_constant_schedule(
        init_value=LR_W,
        boundaries_and_scales={
            20 * steps_per_epoch: 0.2,
            40 * steps_per_epoch: 0.2,
        },
    )
    optim_w = pxu.Optim(lambda: optax.adamw(schedule_w, weight_decay=WEIGHT_DECAY_W), pxu.M(pxnn.LayerParam)(model))

    # Warmup once to compile/shape-init before timings.
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        x0 = jnp.zeros((TRAIN_BATCH_SIZE, 3, 32, 32))
        y0 = jnp.zeros((TRAIN_BATCH_SIZE, N_CLASSES))
        x_inj = jax.vmap(lambda x_i: model.gn_in(model.input_conv(x_i)))(x0)
        model.x_inj_cache.set(x_inj)
        model.vode_z.h.set(jnp.zeros_like(x_inj))
        model.vode_out.h.set(y0)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        train_epoch(train_dl, model=model, optim_w=optim_w, optim_h=optim_h)
        train_time = time.perf_counter() - t0

        train_acc, test_acc, eval_time = evaluate_epoch(
            train_dl,
            test_dl,
            T_STEPS,
            model=model,
            optim_h=optim_h,
            train_eval_samples=TRAIN_EVAL_SAMPLES,
        )

        print(
            f"Epoch {epoch}/{N_EPOCHS} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Train Time: {train_time:.2f}s | "
            f"Eval Time: {eval_time:.2f}s"
        )


if __name__ == "__main__":
    main()
