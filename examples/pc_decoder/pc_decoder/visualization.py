import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import wandb
from matplotlib import axes, figure  # type: ignore
from numpy.typing import NDArray
from pc_decoder.params import Params
from umap import UMAP  # type: ignore

# Use Agg backend for matplotlib so it doesn't open windows
matplotlib.use("agg")


def restore_image(
    image_array: jax.Array, train_data_mean: float, train_data_std: float
) -> NDArray[np.uint8]:
    x = np.asarray(image_array)
    x = x.reshape(28, 28)
    # x = (x - x.min() + 1e-6) / (x.max() - x.min() + 1e-6)  # scale values to [0, 1]
    # low_bound = jnp.percentile(x, 1)
    # high_bound = jnp.percentile(x, 99)
    # x = (x - low_bound + 1e-6) / (high_bound - low_bound + 1e-6)
    x = x * train_data_std + train_data_mean
    # Make sure values are in [0, 1] so we don't get integer overflow on negative values when transforming into uint8.
    x = np.clip(x, 0, 1)
    x = x * 255
    x = x.astype(np.uint8)
    return x


def visualize_predictions(
    *,
    out_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    examples: jax.Array,
    predictions: jax.Array,
    labels: jax.Array,
    params: Params,
    epoch: int,
    train_data_mean: float,
    train_data_std: float,
) -> None:
    logging.info(
        f"Visualizing predictions, up to {params.visualize_n_images_per_label} per label..."
    )
    plt.clf()
    fig, axes = plt.subplots(1, 2)
    visualized_labels: dict[int, int] = defaultdict(int)
    for i, (example, prediction, label) in enumerate(
        zip(examples, predictions, labels)
    ):
        label = label.item()
        if visualized_labels[label] >= params.visualize_n_images_per_label:
            continue
        visualized_labels[label] += 1
        plot_exmaple_and_prediction(
            fig=fig,
            axes=axes,
            example=example,
            prediction=prediction,
            label=label,
            example_index=i,
            out_dir=out_dir,
            run=run,
            epoch=epoch,
            train_data_mean=train_data_mean,
            train_data_std=train_data_std,
        )
        plt.cla()

    plt.clf()
    logging.info(f"Visualized {sum(visualized_labels.values())} examples.")


def plot_exmaple_and_prediction(
    *,
    fig: figure.Figure,
    axes: Iterable[axes.Axes],
    example: jax.Array,
    prediction: jax.Array,
    label: int,
    example_index: int,
    out_dir: Path,
    run,
    epoch: int,
    train_data_mean: float,
    train_data_std: float,
    filename: str | None = None,
) -> None:
    mse = jnp.mean((prediction - example) ** 2).item()

    axes[0].imshow(  # type: ignore
        restore_image(example, train_data_mean, train_data_std),
        cmap="gray",
    )
    axes[0].set_title("Original")  # type: ignore
    axes[1].imshow(  # type: ignore
        restore_image(prediction, train_data_mean, train_data_std),
        cmap="gray",
    )
    axes[1].set_title("Prediction")  # type: ignore
    # Set figure title
    fig.suptitle(f"Epoch {epoch} Label {label} Example {example_index} MSE {mse}")
    if filename is None:
        filename = f"label_{label}_example_{example_index}.png"
    image_path = str(out_dir / filename)
    fig.savefig(image_path)  # type: ignore
    if run is not None:
        run.log({filename: wandb.Image(image_path)})


def visualize_internal_states_clustering(
    *,
    out_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    internal_states: jax.Array,
    labels: jax.Array,
    epoch: int,
) -> None:
    logging.info("Visualizing internal states clustering...")
    # TODO: configure UMAP
    reduced_internal_states = UMAP().fit_transform(jnp.asarray(internal_states))

    internal_states_data: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "label": [],
    }
    internal_states_data["x"].extend(reduced_internal_states[:, 0].tolist())
    internal_states_data["y"].extend(reduced_internal_states[:, 1].tolist())
    internal_states_data["label"].extend(labels.tolist())
    internal_states_df = pd.DataFrame(internal_states_data).sort_values("label")

    plt.clf()
    sns.scatterplot(data=internal_states_df, x="x", y="y", hue="label")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Internal representations of classes. Epoch {epoch}")
    # Only 7/10 labels are present in the legend, but the data is plotted correctly
    plt.legend()
    filename = "pc_decoder_mnist_internal_states.png"
    image_path = str(out_dir / filename)
    plt.savefig(image_path)
    if run is not None:
        run.log({filename: wandb.Image(image_path)})
    plt.clf()
    logging.info("Visualized internal states clustering.")


def create_all_visualizations(
    *,
    out_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    epoch: int,
    examples: jax.Array,
    labels: jax.Array,
    internal_states: jax.Array,
    predictions: jax.Array,
    params: Params,
    train_data_mean: float,
    train_data_std: float,
) -> None:
    logging.info(f"Creating visualizations for epoch {epoch}...")

    generated_dir = out_dir / "generated"
    generated_dir.mkdir()

    visualize_predictions(
        out_dir=generated_dir,
        run=run,
        examples=examples,
        predictions=predictions,
        labels=labels,
        params=params,
        epoch=epoch,
        train_data_mean=train_data_mean,
        train_data_std=train_data_std,
    )
    visualize_internal_states_clustering(
        out_dir=out_dir,
        run=run,
        internal_states=internal_states,
        labels=labels,
        epoch=epoch,
    )
    logging.info(f"Created visualizations for epoch {epoch}.")
