import re
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from pc_decoder.params import Params
from ray import tune

simple_param_name_pattern = re.compile(
    r"\(PCDecoder\)\.(?P<param>[^\.]+)\.\(\"nn\.\(GetAttrKey\(name='(?P<p_type>[^\.]+)'\),\)\",\)"
)
deep_param_name_pattern = re.compile(
    r"\(PCDecoder\)\.(?P<group>[^\.]+)\.\(SequenceKey\(idx=(?P<level>\d+)\), "
    r"(?:\"nn\.\(GetAttrKey\(name='(?P<w_type>[^']+)'\),\)\"\)|'(?P<x_type>[^']+)')"
)

param_type_shortcuts = {
    "weight": "w",
    "bias": "b",
    "x": "x",
}


def init_wandb(
    *,
    params: Params,
    results_dir: Path,
) -> wandb.wandb_sdk.wandb_run.Run | None:
    run_name = (
        f"{params.experiment_name}--{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    if params.do_hypertunning:
        run_name += f"--{tune.get_trial_id()}"  # type: ignore
    params_update = {
        "results_dir": str(results_dir.absolute()),  # type: ignore
    }
    if params.do_hypertunning:
        params_update["trial_id"] = tune.get_trial_id()  # type: ignore
        params_update["trial_name"] = tune.get_trial_name()  # type: ignore
        params_update["trial_dir"] = tune.get_trial_dir()  # type: ignore
    run = wandb.init(
        project="pc-decoder",
        group=params.experiment_name,
        name=run_name,
        tags=["predictive-coding", "autoencoder", "decoder"],
        config={
            **params.dict(),
            **params_update,
        },
    )
    wandb.define_metric("energy/*", step_metric="t_step")
    wandb.define_metric("grad/*", step_metric="t_step")
    wandb.define_metric("train_mse", step_metric="epochs")
    wandb.define_metric("test_mse", step_metric="epochs")
    wandb.define_metric("internal_states_mean", step_metric="epochs")
    wandb.define_metric("internal_states_std", step_metric="epochs")
    return run


def log_train_t_step_metrics(
    *,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    t_step: int,
    energies: list[list[jax.Array]],
    gradients: list[dict[str, jax.Array]],
    params: Params,
) -> None:
    if run is None:
        return
    assert len(energies) == len(gradients) == params.T
    total_energy = jnp.asarray(energies).sum(axis=1).tolist()
    for t, (t_energies, t_gradients) in enumerate(zip(energies, gradients)):
        t_metrics = {
            "t_step": t_step + t,
            "energy/total": total_energy[t],
            "grad/total.sum": np.sum(
                [jnp.abs(x).sum().item() for x in t_gradients.values()]
            ),
            "grad/total.mean": np.mean(
                [jnp.abs(x).mean().item() for x in t_gradients.values()]
            ),
        }
        for node_index, energy in enumerate(t_energies):
            t_metrics[f"energy/node_{node_index}"] = energy.item()
        for long_param_name, param_grad in t_gradients.items():
            m = deep_param_name_pattern.match(long_param_name)
            if m is not None:
                group = m.group("group")
                level = m.group("level")
                w_type = m.group("w_type")
                x_type = m.group("x_type")
                param_type = param_type_shortcuts[w_type or x_type]
                param_name = f"{group}[{level}].{param_type}"
            else:
                m = simple_param_name_pattern.match(long_param_name)
                if m is None:
                    raise RuntimeError(f"Could not parse param name {long_param_name}")
                p_name = m.group("param")
                p_type = m.group("p_type")
                param_type = param_type_shortcuts[p_type]
                param_name = f"{p_name}.{param_type}"
            if param_type == "b":
                continue

            t_metrics[f"grad/{param_name}.sum"] = jnp.sum(jnp.abs(param_grad))
            t_metrics[f"grad/{param_name}.mean"] = jnp.mean(jnp.abs(param_grad))
            t_metrics[f"grad/{param_name}.std"] = jnp.std(param_grad)

        run.log(t_metrics)
