import sys
import os
import multiprocessing as mp

import optuna

SEED = 0
N_CLASSES = 10
N_CHANNELS = 48
N_INNER = 64

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 1000
DATA_ROOT = "~/tmp/cifar10/"

N_EPOCHS_SEARCH = 50
N_TRIALS = 50
N_GPU_WORKERS = 2
STOP_GRAD_F = True


def load_runtime_deps():
    import jax
    import jax.numpy as jnp
    import optax

    import pcx as px
    import pcx.nn as pxnn
    import pcx.predictive_coding as pxc
    import pcx.utils as pxu

    from deq.deq_pc_eval_minimal import evaluate_accuracy
    from deq.deq_pc_minimal_core import DEQPCModel, get_dataloaders, train_on_batch

    return {
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "px": px,
        "pxnn": pxnn,
        "pxc": pxc,
        "pxu": pxu,
        "evaluate_accuracy": evaluate_accuracy,
        "DEQPCModel": DEQPCModel,
        "get_dataloaders": get_dataloaders,
        "train_on_batch": train_on_batch,
    }


def assert_gpu_backend(jax):
    backend = jax.default_backend()
    if backend != "gpu":
        devices = ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices())
        print(
            f"GPU required but current JAX backend is '{backend}'. Available devices: {devices}",
            file=sys.stderr,
        )
        sys.exit(1)


def train_epoch(train_dl, T_steps: int, *, model, optim_w, optim_h, deps):
    jax = deps["jax"]
    train_on_batch = deps["train_on_batch"]
    for x, y in train_dl:
        train_on_batch(
            T_steps,
            x.numpy(),
            jax.nn.one_hot(y.numpy(), N_CLASSES),
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
        )


def make_objective(train_dl, test_dl, *, n_epochs: int, batch_size: int, deps, chan: int = N_CHANNELS):
    jax = deps["jax"]
    jnp = deps["jnp"]
    optax = deps["optax"]
    px = deps["px"]
    pxnn = deps["pxnn"]
    pxc = deps["pxc"]
    pxu = deps["pxu"]
    evaluate_accuracy = deps["evaluate_accuracy"]
    DEQPCModel = deps["DEQPCModel"]


    if STOP_GRAD_F:

        baseline = dict(
            T_train=300,
            nudging=0.1,
            lr_w=0.001,
            wd_w=0.03,
            lr_h=0.2,
            mom_h=0.5,
            init_scale=0.005,
        )

    else:

        baseline = dict(
            T_train=250,
            nudging=0.025,
            lr_w=0.001,
            wd_w=0.03,
            lr_h=0.2,
            mom_h=0.9,
            init_scale=0.00001,
        )

    def objective(trial: optuna.Trial) -> float:
        px.RKG.seed(SEED)

        T_train = trial.suggest_int("T_train", 250, 350)
        nudging = trial.suggest_float("nudging", 0.05, 0.25, log=True)
        lr_w = trial.suggest_float("lr_w", 0.00025, 0.0025, log=True)
        wd_w = trial.suggest_float("wd_w", 0.001, 0.05, log=True)
        lr_h = trial.suggest_float("lr_h", 0.05, 0.5, log=True)
        mom_h = trial.suggest_float("mom_h", 0.25, 0.75, log=True)
        init_scale = trial.suggest_float("init_scale", 0.00005, 0.005, log=True)

        T_eval = T_train

        model = DEQPCModel(
            n_channels=chan,
            n_inner=N_INNER,
            n_classes=N_CLASSES,
            nudging=nudging,
            init_scale=init_scale,
            stop_grad_f=STOP_GRAD_F,
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
            train_epoch(train_dl, T_train, model=model, optim_w=optim_w, optim_h=optim_h, deps=deps)
            acc = evaluate_accuracy(test_dl, T_eval, model=model, optim_h=optim_h)

            best = max(best, acc)
            trial.report(acc, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best

    return objective, baseline


def run_worker(local_gpu: int, n_trials_worker: int, storage: str, study_name: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
    deps = load_runtime_deps()
    jax = deps["jax"]
    px = deps["px"]
    get_dataloaders = deps["get_dataloaders"]

    assert_gpu_backend(jax)
    px.RKG.seed(SEED + local_gpu)

    train_dl, test_dl = get_dataloaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, root=DATA_ROOT)
    objective, baseline = make_objective(
        train_dl,
        test_dl,
        n_epochs=N_EPOCHS_SEARCH,
        batch_size=TRAIN_BATCH_SIZE,
        deps=deps,
        chan=N_CHANNELS,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED + local_gpu, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    if local_gpu == 0 and len(study.trials) == 0:
        study.enqueue_trial(baseline)

    study.optimize(objective, n_trials=n_trials_worker, gc_after_trial=True, show_progress_bar=False)


def main():
    storage = "sqlite:///deqpc_optuna.db"
    study_name = "deqpc_cifar10_long_stopgradient_50_epochs"

    workers = N_GPU_WORKERS
    trials_per_worker = [N_TRIALS // workers] * workers
    for i in range(N_TRIALS % workers):
        trials_per_worker[i] += 1

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id in range(workers):
        p = ctx.Process(
            target=run_worker,
            args=(gpu_id, trials_per_worker[gpu_id], storage, study_name),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode if p.exitcode is not None else 1

    if exit_code != 0:
        for p in procs:
            if p.is_alive():
                p.terminate()
        sys.exit(exit_code)

    study = optuna.load_study(study_name=study_name, storage=storage)
    print("\nBest value (accuracy):", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
