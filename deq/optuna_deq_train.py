import sys

import jax
import jax.numpy as jnp
import optax
import optuna

import pcx as px
import pcx.nn as pxnn
import pcx.predictive_coding as pxc
import pcx.utils as pxu

from deq.deq_pc_eval_minimal import evaluate_accuracy
from deq.deq_pc_minimal_core import DEQPCModel, get_dataloaders, train_on_batch


SEED = 0
N_CLASSES = 10
N_CHANNELS = 48
N_INNER = 64

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
DATA_ROOT = "~/tmp/cifar10/"

N_EPOCHS_SEARCH = 10
N_TRIALS = 150


def train_epoch(train_dl, T_steps: int, *, model, optim_w, optim_h):
    for x, y in train_dl:
        train_on_batch(
            T_steps,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )


def make_objective(train_dl, test_dl, *, n_epochs: int, batch_size: int, chan: int = N_CHANNELS):
    baseline = dict(
        T_train=150,
        nudging=0.05,
        lr_w=0.0025,
        wd_w=0.005,
        lr_h=0.25,
        mom_h=0.5,
        init_scale=0.01,
    )

    def objective(trial: optuna.Trial) -> float:
        px.RKG.seed(SEED)

        T_train = trial.suggest_int("T_train", 80, 300)
        nudging = trial.suggest_float("nudging", 0.001, 0.05, log=True)
        lr_w = trial.suggest_float("lr_w", 0.0005, 0.005, log=True)
        wd_w = trial.suggest_float("wd_w", 0.0001, 0.1, log=True)
        lr_h = trial.suggest_float("lr_h", 0.1, 1.0, log=True)
        mom_h = trial.suggest_float("mom_h", 0.0, 0.9)
        init_scale = trial.suggest_float("init_scale", 0.001, 0.05, log=True)

        T_eval = T_train

        model = DEQPCModel(
            n_channels=chan,
            n_inner=N_INNER,
            n_classes=N_CLASSES,
            nudging=nudging,
            init_scale=init_scale,
        )

        optim_h = pxu.Optim(lambda: optax.sgd(lr_h, momentum=mom_h, nesterov=True))

        steps_per_epoch = len(train_dl)
        schedule_w = optax.piecewise_constant_schedule(
            init_value=lr_w,
            boundaries_and_scales={
                20 * steps_per_epoch: 0.2,
                40 * steps_per_epoch: 0.2,
            },
        )
        optim_w = pxu.Optim(lambda: optax.adamw(schedule_w, weight_decay=wd_w), pxu.M(pxnn.LayerParam)(model))

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            x0 = jnp.zeros((batch_size, 3, 32, 32))
            y0 = jnp.zeros((batch_size, N_CLASSES))
            x_inj = jax.vmap(lambda x_i: model.gn_in(model.input_conv(x_i)))(x0)
            model.x_inj_cache.set(x_inj)
            model.vode_z.h.set(jnp.zeros_like(x_inj))
            model.vode_out.h.set(y0)

        best = 0.0
        for epoch in range(1, n_epochs + 1):
            train_epoch(train_dl, T_train, model=model, optim_w=optim_w, optim_h=optim_h)
            acc = evaluate_accuracy(test_dl, T_eval, model=model, optim_h=optim_h)

            best = max(best, acc)
            trial.report(acc, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best

    return objective, baseline


def main():
    px.RKG.seed(SEED)
    #assert_gpu_backend()

    train_dl, test_dl = get_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, root=DATA_ROOT)

    objective, baseline = make_objective(
        train_dl,
        test_dl,
        n_epochs=N_EPOCHS_SEARCH,
        batch_size=TRAIN_BATCH_SIZE,
        chan=N_CHANNELS,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    storage = "sqlite:///deqpc_optuna_new.db"
    study_name = "deqpc_cifar10"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        study.enqueue_trial(baseline)

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, show_progress_bar=True)

    print("\nBest value (accuracy):", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
