import os
import time
import datetime
import random
from collections import namedtuple

import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch


Dataset = namedtuple("Dataset", ["train_loader", "val_loader", "test_loader"])

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
def add_label_noise(dataset, noise_level=0.2):
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

# Function to get the dataloaders
def get_dataloaders(dataset_name, train_subset_size, batch_size, noise_level=0.2):
    if dataset_name.lower() == "mnist":
        ds = datasets.MNIST
    else:
        raise NotImplementedError(f"Dataset {dataset_name} isn't available")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1).numpy())  # Flatten the image to a vector
    ])

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

    test_set = ds(root='./data', download=True, train=False, transform=transform)
    test_loader = TorchDataloader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    return Dataset(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


class Progress:
    def __init__(self, num_models, num_epochs, num_batches, run_from_term):
        self._num_models = num_models
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._run_from_term = run_from_term
        self._model = 0
        self._epoch = 0
        self._batch = 0
        self._val_loss = 0
        self._train_loss = 0
        self._start_time_global = time.time()
        self._start_time_model = time.time()
        self._time_per_epoch = 0
        self._eta = "Estimating..."
        self._progress_string = "\rNet/Epoch/Batch: {}/{:4d}/{:3d} | {:.2f} % | {} | Train/val: {:.2f}/{:.2f} |"
    

    def init_print(self, num_models, model_name, dataset_name):
        if self._run_from_term:
            _, width = os.popen("stty size", "r").read().split()
        else:
            width = 60
        print("=" * int(width))
        print("Training {} {} models on {} dataset...".format(
            num_models, model_name, dataset_name))

        print("=" * int(width))


    def finished_model(self, num_parameters, test_loss, acc):
        if self._run_from_term:
            _, width = os.popen("stty size", "r").read().split()
        else:
            width = 60
        finished_string = "=" * int(width)
        finished_string += "\nFinished training model with {} parameters".format(
            num_parameters)
        finished_string += "\nFinal train/val/test loss: {:.2f}/{:.2f}/{:.2f}".format(
            self._train_loss, self._val_loss, test_loss)
        finished_string += "\nAccuracy: {:.3}".format(acc)
        training_time = datetime.timedelta(
            seconds=round(time.time()-self._start_time_model))
        finished_string += "\nTraining model took: {}\n".format(training_time)
        print(finished_string)

    def finished_training(self):
        if self._run_from_term:
            _, width = os.popen("stty size", "r").read().split()
        else:
            width = 60
        finished_string = "=" * int(width)
        finished_string += "\nFinished entire training of {} models.".format(
            self._num_models)
        training_time = datetime.timedelta(
            seconds=round(time.time()-self._start_time_global))
        finished_string += "\n Training took {}\n".format(training_time)
        finished_string += "=" * int(width)
        print(finished_string)

    def update_batch(self):
        self._batch += 1
        if self._run_from_term:
            self._print_progress_string()

    def update_epoch(self, train_loss, val_loss):
        self._batch = 0
        self._epoch += 1
        if self._epoch == 1:
            self._time_per_epoch = time.time() - self._start_time_model
        secs_left = self._time_per_epoch * (self._num_epochs - self._epoch)
        self._eta = datetime.timedelta(seconds=round(secs_left))
        self._val_loss = val_loss
        self._train_loss = train_loss
        if self._run_from_term:
            self._print_progress_string()

    def update_model(self):
        self._model += 1
        self._epoch = 0
        if self._model == 0:
            self._start_time_global = time.time()
        self._start_time_model = time.time()
        if self._run_from_term:
            self._print_progress_string()

    def _print_progress_string(self):
        # percent epochs = current_epoch / num_epochs
        # percent batches = current batch / num_batches
        # percent total = current_batch_total / num_batches_total
        # num_batches_total = num_epochs * num_batches
        # current_batch_total = (current_epoch-1) * num_batches + current_batches
        percentage_done = 100 * (self._epoch * self._num_batches +
                                 self._batch)/(self._num_epochs*self._num_batches)
        current_progress_string = self._progress_string.format(self._model,
                                                               self._epoch,
                                                               self._batch,
                                                               percentage_done,
                                                               self._eta,
                                                               self._train_loss,
                                                               self._val_loss)
        if self._run_from_term:
            _, width = os.popen("stty size", "r").read().split()
        else:
            width = 60
        available_width = max(
            0, int(width) - len(current_progress_string.expandtabs()) - 2)

        progress_width = int(available_width * percentage_done/100)
        progress_bar = "[" + "#" * progress_width + \
            " " * (available_width - progress_width) + "]"
        current_progress_string += progress_bar
        print(current_progress_string, end="\r")
