import os
import time
import datetime
import random
import pickle
from typing import NamedTuple

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import jax
import jax.numpy as jnp

def set_random_seed(seed):
    # Set the seed for reproducibility    
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)

# Function to calculate the mean and std of a dataset
def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)
    data = next(iter(loader))[0]
    mean = data.mean()
    std = data.std()
    return mean.item(), std.item()

# This is a simple collate function that stacks numpy arrays used to interface
# the PyTorch dataloader with JAX. In the future we hope to provide custom dataloaders
# that are independent of PyTorch.
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

# The dataloader assumes cuda is being used, as such it sets 'pin_memory = True' and
# 'prefetch_factor = 2'. Note that the batch size should be constant during training, so
# we set 'drop_last = True' to avoid having to deal with variable batch sizes.
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=16,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

# Function to add noise to the labels in the dataset
def add_label_noise(dataset, noise_level=0.0):
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    num_noisy = int(noise_level * len(targets))
    noisy_indices = np.random.choice(len(targets), num_noisy, replace=False)

    for idx in noisy_indices:
        original_label = targets[idx]
        new_label = original_label
        while new_label == original_label:
            new_label = np.random.randint(0, num_classes)
        targets[idx] = new_label

    dataset.targets = torch.tensor(targets)
    return dataset

# Function to filter the first 10 classes of EMNIST Letters and adjust targets to 0-9
def filter_first_10_classes(dataset):
    valid_classes = list(range(1, 11))  # classes 1 to 10
    mask = torch.isin(dataset.targets, torch.tensor(valid_classes))
    dataset.targets = dataset.targets[mask] - 1  # Adjust targets to range 0-9, so as to match MNIST and KMNIST
    dataset.data = dataset.data[mask]
    return dataset

# Define transform functions at the top level
def flatten_image(x):
    return x.view(-1).numpy()

# Custom Dataset class to include additional attributes
class CustomDataset(NamedTuple):
    train_loader: TorchDataloader
    val_loader: TorchDataloader
    test_loader: TorchDataloader
    classes: list
    class_to_idx: dict

# Custom NotMNIST dataset class
class NotMNIST(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self._load_images(images_path)
        self.labels = self._load_labels(labels_path)
        self.transform = transform

        # only keep 8000 samples for now (b/c EMNIST is 8000 samples only in test set)
        self.images = self.images[:8000]
        self.labels = self.labels[:8000]

    def _load_images(self, path):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).copy()  # Ensure the array is writable

    def _load_labels(self, path):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.copy()  # Ensure the array is writable

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to get the dataloaders
def get_dataloaders(dataset_name, train_subset_size, batch_size, noise_level=0.0):
    if dataset_name.lower() == "mnist":
        ds = datasets.MNIST

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean and std for normalization
            transforms.Lambda(flatten_image)  # Flatten the image to a vector
        ])
    elif dataset_name.lower() == "fmnist":
        ds = datasets.FashionMNIST

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)), # FashionMNIST mean and std for normalization
            transforms.Lambda(flatten_image)  # Flatten the image to a vector
        ])
    elif dataset_name.lower() == "emnist":
        ds = datasets.EMNIST

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1722,), (0.3309,)), # EMNIST letters mean and std for normalization
            transforms.Lambda(flatten_image)  # Flatten the image to a vector
        ])
    elif dataset_name.lower() == "kmnist":
        ds = datasets.KMNIST

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1918,), (0.3483,)), # KMNIST mean and std for normalization
            transforms.Lambda(flatten_image)  # Flatten the image to a vector
        ])
    # NOTE: notMNIST dataset is not available for training, only for testing
    elif dataset_name.lower() == "notmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4248,), (0.4585,)),  # notMNIST mean and std for normalization
            transforms.Lambda(flatten_image)  # Flatten the image to a vector
        ])
        
        # Load the notMNIST dataset using the custom dataset class
        notmnist_dataset = NotMNIST(
            images_path='./data/notMNIST/raw/t10k-images-idx3-ubyte',
            labels_path='./data/notMNIST/raw/t10k-labels-idx1-ubyte',
            transform=transform
        )

        dataloader = TorchDataloader(notmnist_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

        return CustomDataset(
            train_loader=dataloader,
            val_loader=dataloader,
            test_loader=dataloader,
            classes=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            class_to_idx={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} isn't available")

    # Train set
    if dataset_name.lower() == "emnist":
        train_set = ds(root='./data', split='letters', download=True, train=True, transform=transform)
        train_set = filter_first_10_classes(train_set)
        # also remove 'N/A' key from classes and class_to_idx
        train_set.classes = train_set.classes[1:11] # classes 1 to 10 for EMNIST letters a-j
        # remove all keys not in classes 1 to 10
        for key in list(train_set.class_to_idx.keys()):
            if key not in train_set.classes[1:11]:
                train_set.class_to_idx.pop(key)

    else:
        train_set = ds(root='./data', download=True, train=True, transform=transform)

    train_set = add_label_noise(train_set, noise_level=noise_level)

    val_subset_size = int(0.2 * train_subset_size)
    random_train_indices = np.random.choice(len(train_set), size=train_subset_size, replace=False)
    remaining_indices = np.setdiff1d(np.arange(len(train_set)), random_train_indices)
    random_val_indices = np.random.choice(remaining_indices, size=val_subset_size, replace=False)

    train_loader = TorchDataloader(
        train_set, batch_size=batch_size, num_workers=16,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(random_train_indices))

    val_loader = TorchDataloader(
        train_set, batch_size=batch_size, num_workers=16,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(random_val_indices))

    # Test set
    if dataset_name.lower() == "emnist":
        test_set = ds(root='./data', split='letters', download=True, train=False, transform=transform)
        test_set = filter_first_10_classes(test_set)
    else:
        test_set = ds(root='./data', download=True, train=False, transform=transform)

    test_loader = TorchDataloader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    # Create the custom dataset
    custom_dataset = CustomDataset(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=train_set.classes,
        class_to_idx=train_set.class_to_idx
    )

    return custom_dataset


# Function to creat an equivalent data_loader with only different batch size
def create_new_test_loader_with_batch_size(test_loader, new_batch_size):
    """
    Create a new DataLoader based on the attributes of the passed test_loader
    with a different batch size.

    Parameters:
    - test_loader: The original DataLoader
    - new_batch_size: The desired batch size for the new DataLoader

    Returns:
    - new_test_loader: A new DataLoader with the specified batch size
    """
    # Extract the dataset from the original DataLoader
    dataset = test_loader.dataset

    # Create a new DataLoader with the new batch size
    new_test_loader = DataLoader(
        dataset,
        batch_size=new_batch_size,
        shuffle=False,  # Typically, test_loader should not shuffle data
        num_workers=test_loader.num_workers,
        collate_fn=test_loader.collate_fn,
        pin_memory=test_loader.pin_memory,
        drop_last=True,
        timeout=test_loader.timeout,
        worker_init_fn=test_loader.worker_init_fn,
        multiprocessing_context=test_loader.multiprocessing_context,
        generator=test_loader.generator,
        prefetch_factor=test_loader.prefetch_factor,
        persistent_workers=test_loader.persistent_workers
    )

    return new_test_loader


class DatasetProcessorData:
    def __init__(self, dataset_name, T_inf, new_batch_size):
        self.dataset_name = dataset_name
        self.T_inf = T_inf
        self.new_batch_size = new_batch_size

        self.X_test_jax = None
        self.ys_test_jax = None
        self.logits_test_jax = None
        self.max_softmax_values = None
        self.e_pre = None
        self.e_post = None
        self.p_pre = None
        self.p_post = None
        self.NLL_pre = None
        self.likelihood_pre = None
        self.NLL_post = None
        self.likelihood_post = None
