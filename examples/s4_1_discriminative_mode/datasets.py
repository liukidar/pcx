
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tinyimagenet import TinyImageNet

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
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


def get_dataloaders_cifar10(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # These are normalisation factors found online.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        # These are normalisation factors found online.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        "~/tmp/cifar10/",
        transform=t_val,
        download=True,
        train=False,
    )

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, test_dataloader

def get_dataloaders_fmnist(batch_size: int):
    t = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize with mean and std dev
    transforms.Lambda(lambda x: torch.flatten(x)),  # Flatten the tensor
    lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fmnist/",
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fmnist/",
        transform=t,
        download=True,
        train=False,
    )
        
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader

def get_dataloaders_mnist(batch_size: int):
    t = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize with mean and std dev
    transforms.Lambda(lambda x: torch.flatten(x)),  # Flatten the tensor
    lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.MNIST(
        "~/tmp/mnist/",
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_dataset = torchvision.datasets.MNIST(
        "~/tmp/mnist/",
        transform=t,
        download=True,
        train=False,
    )
        
    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, test_dataloader

def get_dataloaders_cifar100(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # These are normalisation factors found online.
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        # These are normalisation factors found online.
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        "~/tmp/cifar100/",
        transform=t,
        download=True,
        train=True,
    )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_dataset = torchvision.datasets.CIFAR100(
        "~/tmp/cifar100/",
        transform=t_val,
        download=True,
        train=False,
    )

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, test_dataloader

def get_dataloaders_tinyimagenet(batch_size: int):
    t_val = transforms.Compose([
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(mean=TinyImageNet.mean, std=TinyImageNet.std),
        lambda x: x.numpy()
    ])

    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(56),
        transforms.ToTensor(),
        transforms.Normalize(mean=TinyImageNet.mean, std=TinyImageNet.std),
        lambda x: x.numpy()
    ])

    train_dataset = TinyImageNet(
        "~/tmp/tinyimagenet/", 
        split="train", 
        transform=t
        )

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_dataset = TinyImageNet(
        "~/tmp/tinyimagenet/", 
        split="val",
        transform=t_val
        )

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, test_dataloader

def get_loader(dn, batch_size):
    if dn == 'MNIST':
        return get_dataloaders_mnist(batch_size)
    elif dn == 'FashionMNIST':
        return get_dataloaders_fmnist(batch_size)
    elif dn == 'CIFAR10':
        return get_dataloaders_cifar10(batch_size)
    elif dn == 'CIFAR100':
        return get_dataloaders_cifar100(batch_size)
    elif dn == 'TinyImageNet':
        return get_dataloaders_tinyimagenet(batch_size)
    else:
        raise ValueError("Invalid dataset name")