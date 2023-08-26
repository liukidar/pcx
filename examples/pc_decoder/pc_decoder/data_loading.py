import logging
from typing import Any

import jax
import jax.numpy as jnp
import torch
from pc_decoder.params import Params
from torch.utils.data import DataLoader  # type: ignore
from torchvision import datasets, transforms


def download_datasets(params: Params):
    datasets.MNIST(root=params.data_dir, train=True, download=True)  # type: ignore
    datasets.MNIST(root=params.data_dir, train=False, download=True)  # type: ignore


def get_data_loaders(
    params: Params,
) -> tuple[DataLoader[Any], DataLoader[Any], float, float]:
    train_dataset = datasets.MNIST(root=params.data_dir, train=True)

    train_data = train_dataset.data / 255
    train_data_mean = torch.mean(train_data).item()
    train_data_std = torch.std(train_data).item()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((train_data_mean,), (train_data_std,)),
        ]
    )

    train_dataset = datasets.MNIST(root=params.data_dir, train=True, transform=transform)  # type: ignore
    test_dataset = datasets.MNIST(root=params.data_dir, train=False, transform=transform)  # type: ignore

    assert (
        len(train_dataset) % params.batch_size
        == len(test_dataset) % params.batch_size
        == 0
    ), "All batches must have the same size!"

    def collate_into_jax_arrays(
        examples: list[tuple[torch.Tensor, Any]]
    ) -> tuple[jax.Array, list[Any]]:
        data = jnp.array([x[0].numpy() for x in examples]).reshape(len(examples), -1)
        targets = [x[1] for x in examples]
        return data, targets

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=collate_into_jax_arrays,
        # drop_last=True,
    )

    return train_loader, test_loader, train_data_mean, train_data_std


def get_stratified_test_batch(
    test_loader: DataLoader[Any],
) -> tuple[jax.Array, jax.Array]:
    num_classes = 10
    batch_size = test_loader.batch_size
    assert batch_size is not None
    num_per_label = batch_size // num_classes
    additional_examples = max(0, batch_size - num_per_label * num_classes)

    selected_indexes: dict[int, list[int]] = {k: [] for k in range(num_classes)}
    label: int
    for index, (_, label) in enumerate(test_loader.dataset):  # type: ignore
        selected_indexes[label].append(index)
    for label in selected_indexes:
        additional_examples += max(0, num_per_label - len(selected_indexes[label]))
    for label in selected_indexes:
        selected_indexes[label] = selected_indexes[label][
            : num_per_label + additional_examples
        ]
        if len(selected_indexes[label]) > num_per_label:
            additional_examples = 0
    for label, indexes in selected_indexes.items():
        if not indexes:
            logging.warning(f"No visualization examples for label {label}!")
    selected_examples = []
    selected_labels = []
    for label, indexes in selected_indexes.items():
        for index in indexes:
            selected_examples.append(test_loader.dataset[index][0])
            assert test_loader.dataset[index][1] == label
            selected_labels.append(label)

    examples = jnp.array([x.reshape(-1).numpy() for x in selected_examples])
    labels = jnp.array(selected_labels)
    return examples, labels
