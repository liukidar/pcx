from typing import Callable
import torch
import numpy as np
import torchvision
from tinyimagenet import TinyImageNet
import torchvision.transforms as transforms


# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf


import json
import copy

class ConvNet(pxc.EnergyModule):
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
                pxnn.Linear(512 * 4 * 4, self.nm_classes.get()),
            ),
        ]

        self.vodes = [
            pxc.Vode(shape) for _, shape in zip(range(len(self.feature_layers)), [
                (128, 28, 28),
                (128, 28, 28),
                (256, 14, 14), 
                (256, 12, 12),
                (512, 6, 6),
                (512, 4, 4)
            ])
        ] + [
            pxc.Vode((self.nm_classes.get(),), energy_fn=pxc.ce_energy)]
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, beta: float = 1.0):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")


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
        "~/codes/stune/Exp_new/testti/tinyimagenet/", 
        split="train", 
        transform=t
        )
    
    print(len(train_dataset))

    train_dataloader = TorchDataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_dataset = TinyImageNet(
        "~/codes/stune/Exp_new/testti/tinyimagenet/", 
        split="val",
        transform=t_val
        )

    print(len(test_dataset))

    test_dataloader = TorchDataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, test_dataloader


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: ConvNet, beta=1.0):
    return model(x, y, beta=beta)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: ConvNet):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: ConvNet, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model, beta=beta)
    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"], True)
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"], mul=1/beta)


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: ConvNet):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        outputs = forward(x, None, model=model)
    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, T, *, model: ConvNet, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta
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
        nm_classes=200, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]))
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, 3, 56, 56)), None, model=model)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=run_info["hp/optim/w/lr"],  # 初始学习率
            peak_value=1.1 * run_info["hp/optim/w/lr"],  # 最大学习率
            warmup_steps=0.1 * len(train_dataloader) * nm_epochs,  # 预热步数 
            decay_steps=len(train_dataloader)*nm_epochs,  # 衰减步数
            end_value=0.1 * run_info["hp/optim/w/lr"],  # 最小学习率
            exponent=1.0)

        optim_h = pxu.Optim(
            optax.chain(
                optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
            ),
            pxu.Mask(pxc.VodeParam)(model),
        )
        optim_w = pxu.Optim(optax.adamw(schedule,weight_decay=run_info["hp/optim/w/wd"]), pxu.Mask(pxnn.LayerParam)(model))
    
    
    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []
    for e in range(nm_epochs):
        beta = run_info["hp/beta_factor"] * (run_info["hp/beta"] + run_info["hp/beta_ir"]*e)
        if abs(beta) >= 1.0 :
            beta = 1.0
        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)
        a, a5, y = eval(test_dataloader, model=model)
        accuracies.append(float(a))
        
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
