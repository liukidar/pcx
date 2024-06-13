from typing import Callable
import torch
import numpy as np
import torchvision
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

# stune
import stune
import json


class MLP(pxc.EnergyModule):
    def __init__(self, nm_classes: int, act_fn: Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)

        # Note we use a custom activation function and not exclusively ReLU since
        # it does not seem to perform as well as in backpropagation
        self.act_fn = px.static(act_fn)

        self.layers = [
            (pxnn.Linear(784, 128), self.act_fn),
            (pxnn.Linear(128, 128), self.act_fn),
            (pxnn.Linear(128, self.nm_classes.get()),),
        ]

        # self.long_skip_connection = pxnn.Linear(4096, self.nm_classes.get())

        # We define the Vode modules. Note that currently each vode requires its shape
        # to be manually specified. This will be improved in the near future as lazy
        # initialisation should be possible.
        self.vodes = [pxc.Vode((128,)) for _ in range(len(self.layers) - 1)] + [pxc.Vode((self.nm_classes.get(),))]

        # Remember 'frozen' is a user specified attribute used later in the gradient function
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, beta: float = 1.0):
        x = x.flatten()
        for i, (block, node) in enumerate(zip(self.layers, self.vodes)):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") + beta * (y - self.vodes[-1].get("u")))

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
    t = transforms.Compose([transforms.ToTensor(), lambda x: x.numpy()])

    t_val = transforms.Compose([transforms.ToTensor(), lambda x: x.numpy()])

    train_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
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

    test_dataset = torchvision.datasets.FashionMNIST(
        "~/tmp/fashion-mnist/",
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


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: MLP, beta=1.0):
    return model(x, y, beta=beta)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: MLP):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, x: jax.Array, y: jax.Array, *, model: MLP, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0
):
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
    optim_w.step(model, g["model"], mul=1 / beta)


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: MLP):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: MLP, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h, beta=beta
        )


def eval(dl, *, model: MLP):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def main(run_info: stune.RunInfo):
    intermidiate_savepath = run_info["study"] + ".json" if run_info.study_name else None

    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    model = MLP(nm_classes=10, act_fn=getattr(jax.nn, run_info["hp/act_fn"]))

    train_dataloader, test_dataloader = get_dataloaders(batch_size)

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, 28, 28)), None, model=model)

        optim_h = pxu.Optim(
            optax.chain(
                optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
            )
        )
        optim_w = pxu.Optim(
            optax.adamw(run_info["hp/optim/w/lr"], weight_decay=run_info["hp/optim/w/wd"]),
            pxu.Mask(pxnn.LayerParam)(model),
        )

    best_accuracy = 0
    acc_list = []
    below_times = 0
    stop_increase = 0
    for e in range(nm_epochs):
        beta = run_info["hp/beta_factor"] * (run_info["hp/beta"] + run_info["hp/beta_ir"] * e)
        if abs(beta) >= 1.0:
            beta = 1.0
        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta)
        a, y = eval(test_dataloader, model=model)
        acc_list.append(float(a))
        if run_info.study_name is None:
            print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
        if a > best_accuracy:
            best_accuracy = a
            stop_increase = 0
        else:
            stop_increase += 1
        if float(a) < 0.15 or (e > 10 and float(a) < 0.5):
            below_times += 1
        else:
            below_times = 0
        if below_times >= 5 or (e >= 25 and stop_increase >= 5):
            break

    config_save = run_info.log
    config_save["results"] = acc_list
    if intermidiate_savepath:
        try:
            with open(intermidiate_savepath, "r") as file:
                # load json file if exists
                data = json.load(file)
        except FileNotFoundError:
            data = []
        except json.decoder.JSONDecodeError:
            data = []

        # add new data to the list
        data.append(config_save)

        with open(intermidiate_savepath, "w") as file:
            json.dump(data, file, indent=4)

    return best_accuracy


if __name__ == "__main__":
    run_info = stune.RunInfo(OmegaConf.load("EP.yaml"))
    main(run_info)
