from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf



import json


def se_loss(output, one_hot_label):
    return (jnp.square(output - one_hot_label)).sum()


def ce_loss(output, one_hot_label):
    return -(one_hot_label * jax.nn.log_softmax(output)).sum()


class ConvNet(px.Module):
    def __init__(
        self,
        nm_classes: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (
                pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            )
        ]
        self.classifier_layers = [
            (
                pxnn.Linear(512 * 1 * 1, self.nm_classes.get()),
            ),
        ]


    def __call__(self, x: jax.Array):
        for block in self.feature_layers:
            for layer in block:
                x = layer(x)

        x = x.flatten()
        for block in self.classifier_layers:
            for layer in block:
                x = layer(x)
           
        return x


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


def get_dataloaders(batch_size: int):
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


@pxf.vmap({"model": None}, in_axes=0, out_axes=0)
def forward(x, *, model: ConvNet):
    return model(x)


@pxf.vmap({"model": None}, in_axes=(0, 0), out_axes=(None, 0), axis_name="batch")
def loss(x, y, *, model: ConvNet):
    y_ = model(x)
    return jax.lax.pmean(ce_loss(y_, y), "batch"), y_


@pxf.jit()
def train_on_batch(x: jax.Array, y: jax.Array, *, model: ConvNet, optim_w: pxu.Optim):
    model.train()
    # Learning step
    with pxu.step(model):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)(x, y,  model=model)
    optim_w.step(model, g["model"])
    

@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: ConvNet):
    model.eval()

    with pxu.step(model):
        outputs = forward(x, model=model)

    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, *, model: ConvNet, optim_w: pxu.Optim):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w
        )


def eval(dl, *, model: ConvNet):
    acc = []
    acc5 = []
    ys_ = []

    for x, y in dl:
        a, a5, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        acc5.append(a5)
        ys_.append(y_)

    return np.mean(acc), np.mean(acc5), np.concatenate(ys_)


def main(run_info):
    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    model = ConvNet(
        nm_classes=100, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"])
    )
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader)*nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0)

    optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []
    for e in range(nm_epochs):
        train(train_dataloader, model=model, optim_w=optim_w)
        a, a5, y = eval(test_dataloader, model=model)
        if a > best_accuracy:
            best_accuracy = a
        if a5 > best_accuracy5:
            best_accuracy5 = a5
        accuracies.append(float(a))
        accuracies5.append(float(a5))

    del train_dataloader
    del test_dataloader

    return float(best_accuracy), float(best_accuracy5), accuracies, accuracies5


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import seed5 as seed
    run_info = seed.RunInfo(
        OmegaConf.load(sys.argv[1])
    )
    seed.run(main)(run_info)
