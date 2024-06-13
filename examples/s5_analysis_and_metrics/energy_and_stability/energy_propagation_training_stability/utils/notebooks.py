import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tueplots import bundles
import yaml

from utils.cluster import ClusterManager

cluster = ClusterManager()


def incrase_size(bundle: Dict[str, Any], add: float) -> Dict[str, Any]:
    keys = [
        "font.size",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "axes.titlesize",
    ]
    for key in keys:
        bundle[key] = int(bundle[key] + add)
    return bundle


NEURIPS_FORMAT_FULL = bundles.neurips2024(rel_width=1.0, usetex=False)
NEURIPS_FORMAT_FULL = incrase_size(NEURIPS_FORMAT_FULL, 3)

NEURIPS_FORMAT_HALF = bundles.neurips2024(rel_width=0.5, usetex=False)
NEURIPS_FORMAT_HALF = incrase_size(NEURIPS_FORMAT_HALF, 3)

NEURIPS_FORMAT_THIRD = bundles.neurips2024(rel_width=0.333, usetex=False)
NEURIPS_FORMAT_QUARTER = bundles.neurips2024(rel_width=0.25, usetex=False)

NEURIPS_FORMAT_HALF_TALL = NEURIPS_FORMAT_HALF.copy()
NEURIPS_FORMAT_HALF_TALL["figure.figsize"] = (
    NEURIPS_FORMAT_HALF["figure.figsize"][0],
    NEURIPS_FORMAT_HALF["figure.figsize"][1] * 1.4,
)


chance_performances = {
    "mnist": 0.1,
    "two_moons": 0.5,
    "fashion_mnist": 0.1,
}

linear_performances = {
    "mnist": 0.92,
    "two_moons": 0.86,
    "fashion_mnist": 0.83,
}

record_mappings = {
    "Two Moons": "two_moons",
    "MNIST": "mnist",
    "FashionMNIST": "fashion_mnist",
    "PC": "PC",
    "BP": "BP",
    "SGD": "sgd",
    "ADAMW": "adamw",
    0.0: 0.0,
    0.9: 0.9,
}


def load_data(
    study: str, experiment_ids: str | List[str]
) -> Tuple[str, List[Dict], List[Dict], pd.DataFrame]:
    """This loads the data saved during training using a list of experiment_ids. See notebooks for details."""
    # convert experiment_ids to list
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]
    else:
        experiment_ids = experiment_ids

    # load results
    print(f"Loading experiments: {experiment_ids}")

    results = []
    configs = []
    experiment_folders = []
    missing_results = []
    missing_configs = []

    for experiment_id in experiment_ids:
        experiment_folder = os.path.join(cluster.artifact_dir, study, experiment_id)
        load_path = os.path.join(experiment_folder, "results.pickle")
        try:
            with open(load_path, "rb") as f:
                loaded_result = pickle.load(f)
                results.extend(loaded_result)
                n_results = len(loaded_result)
        except FileNotFoundError:
            missing_results.append(experiment_id)
            n_results = 0

        try:
            with open(os.path.join(experiment_folder, "config.yaml"), "r") as f:
                config = yaml.safe_load(f)
                # config = OmegaConf.load(f)
                configs.extend([config] * n_results)
        except FileNotFoundError:
            missing_configs.append(experiment_id)

        experiment_folders.append(experiment_folder)

    print(f"Missing Experiment Results: {missing_results}")
    print(f"Missing Configs: {missing_configs}")

    # compute metrics in json
    results_with_metrics = []
    for cfg, result in zip(configs, results):
        result_dict = {}
        result_dict["experiment"] = cfg
        result_dict["config"] = result["config"]
        result_dict["results"] = {}
        for metric_name, metric_data in result["results"].items():
            if metric_name == "accuracy":
                result_dict["results"][metric_name] = float(metric_data)
            else:
                dict_per_layer = {
                    f"{metric_name}_{layer_name}": float(layer_values.flatten()[-1])
                    for layer_name, layer_values in metric_data.items()
                }
                # ratios of metrics between layers
                for l in range(len(dict_per_layer) - 1, 0, -1):
                    dict_per_layer[f"{metric_name}_ratio_{l-1}/{l}"] = dict_per_layer[
                        f"{metric_name}_{l-1}"
                    ] / (dict_per_layer[f"{metric_name}_{l}"] + 1e-32)

                result_dict["results"].update(dict_per_layer)

        results_with_metrics.append(result_dict)

    # json to df
    df = pd.json_normalize(results_with_metrics)
    df["id"] = df.index
    df["config.w_lr_log"] = np.log10(df["config.w_lr"])
    df["config.h_lr_log"] = np.log10(df["config.h_lr"])
    df["config.optim"] = (
        df["config.optimizer_w"] + "-" + df["config.momentum_w"].astype(str)
    )
    possible_optimizers = [
        "sgd-0.0",
        "sgd-0.5",
        "sgd-0.9",
        "sgd-0.95",
        "sgd-0.99",
        "adamw-0.9",
    ]
    df["config.optim"] = pd.Categorical(
        df["config.optim"],
        [x for x in possible_optimizers if x in df["config.optim"].values],
        ordered=False,
    )
    df["all_experimental_vars"] = df.apply(
        lambda x: f'{x["config.optim"]}-{x["config.w_lr"]:.5f}-{x["config.h_lr"]:.5f}-{x["config.hidden_dims"]}-{x["config.act_fn"]}',
        axis=1,
    )
    df = df.sort_values("results.accuracy", ascending=False)
    # model-optimizer-optim
    df["condition"] = df.apply(
        lambda x: f'{x["config.model_definition"]}-{x["config.optim"]}', axis=1
    )

    return experiment_folder, results, results_with_metrics, df
