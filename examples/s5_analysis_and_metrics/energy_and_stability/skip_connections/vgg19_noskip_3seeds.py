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
import copy
import os
from data_utils import get_vision_dataloaders, reconstruct_image, seed_everything  # noqa: E402

VGG_types = {
    "CNN": [64, "M", 128, "M", 128, 64, "M"],
    "EP": [128, "M", 256, "M", 512, "M", 512, "M"],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGGNet(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        in_height: int,
        in_width: int,
        in_channels: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers, vodes_feature = self.init_convs(VGG_types['VGG19'], in_channels, in_height, in_width)
        self.classifier_layers, vodes_classifer = self.init_fcs(VGG_types['VGG19'], in_height, in_width, 4096, self.nm_classes.get())
        self.vodes = vodes_feature + vodes_classifer
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array):
        for block, node in zip(self.feature_layers, self.vodes[: len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers) :]):
            for layer in block:
                x = layer(x)
            x = node(x)

        if y is not None:
            self.vodes[-1].set("h", y)

        return self.vodes[-1].get("u")

    def init_convs(self, architecture, in_channels, in_height, in_width):
        layers = []
        vodes = []

        for i in range(len(architecture) - 1):
            x = architecture[i]
            next_x = architecture[i + 1]
            
            if type(x) == int:
                out_channel = x
                if type(next_x) == int:
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                        )
                    )
                    vodes.append(
                        pxc.Vode((out_channel, in_height, in_width))
                    )
                elif next_x == "M":
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                            pxnn.MaxPool2d(kernel_size=2, stride=2)
                        )
                    )
                    in_height = in_height // 2
                    in_width = in_width // 2
                    vodes.append(
                        pxc.Vode((out_channel, in_height, in_width))
                    )
                else:
                    raise ValueError(
                        f"some errors in architecture file"
                    )
                in_channels = x
            else:
                pass
        return layers, vodes
                     
    def init_fcs(self, architecture, in_height, in_width, num_hidden, nm_classes):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (in_height % factor) + (in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = in_height // factor
        out_width = in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        layers = [
            (
                pxnn.Linear(last_out_channels * out_height * out_width, num_hidden),
                self.act_fn,
                # Mark, do we need a DropOut?
            ),
            # (
            #     pxnn.Linear(num_hidden, num_hidden),
            #     self.act_fn,
            # ),
            (
                pxnn.Linear(num_hidden, nm_classes),
                # self.act_softmax
            ),       
        ]
        vodes = [
            pxc.Vode((num_hidden,)),
            # pxc.Vode((num_hidden,)),
            pxc.Vode((nm_classes,), energy_fn=pxc.se_energy)
        ]
        return layers, vodes


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
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.numpy(),
        ]
    )
    t_val = transforms.Compose(
        [
            transforms.ToTensor(),
            # These are normalisation factors found online.
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x.numpy(),
        ]
    )

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
        num_workers=4,
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
        num_workers=4,
    )

    return train_dataloader, test_dataloader


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: VGGNet):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: VGGNet):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_

@pxf.jit(static_argnums=[0,6])
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: VGGNet, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()
    optim_h.init(model)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    # r = []
    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"])
        # r.append(
        #     jnp.stack(
        #         [
        #             jnp.mean(jnp.linalg.norm(jnp.reshape(u.h.get(), (-1, x.shape[0])), axis=-1))
        #             for u in updates.vodes[:-1]
        #         ]
        #     )
        # )

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g1 = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    
    optim_w.step(model, g1["model"])
    # return jnp.stack(r)


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: VGGNet):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: VGGNet, optim_w: pxu.Optim, optim_h: pxu.Optim):
    
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h
        )

def eval(dl, *, model: VGGNet):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)

def main(seed):
    # Hardcoded values from the YAML file
    batch_size = 128
    nm_epochs = 30
    optim_x_lr = 0.5
    optim_x_momentum = 0.5
    T = 24
    study_name = "vgg_no_skip_final10"
    intermidiate_savepath = f"{study_name}_{seed}.json"
    
    seed_everything(seed)

    train_dataloader, test_dataloader = get_dataloaders(batch_size)
    model = VGGNet(
        nm_classes=10, 
        in_channels=3,
        in_height=32,
        in_width=32, 
        act_fn=jax.nn.leaky_relu,
    )

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, 3, 32, 32)), None, model=model)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0001,  # Initial learning rate
            peak_value=1.1 * 0.0001,  # Peak learning rate
            warmup_steps=0.1 * len(train_dataloader) * nm_epochs,  # Warmup steps
            decay_steps=len(train_dataloader) * nm_epochs,  # Decay steps
            end_value=0.1 * 0.0001,  # Minimum learning rate
            exponent=1.0
        )

        optim_h = pxu.Optim(
            optax.chain(
                optax.sgd(optim_x_lr, momentum=optim_x_momentum),
            ),
            pxu.Mask(pxc.VodeParam)(model),
        )
        optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=0.0005), pxu.Mask(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    acc_list = []
    below_times = 0
    for e in range(nm_epochs):
        train(train_dataloader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        a, y = eval(test_dataloader, model=model)
        acc_list.append(float(a))
        print(f"Epoch {e + 1}/{nm_epochs} - Test Accuracy: {a * 100:.2f}%")
        best_accuracy = max(best_accuracy, a)
        # if float(a) < 0.15 or (float(a) < 0.5 and e > 50):
        #     below_times += 1
        # else:
        #     below_times = 0
        # if below_times >= 5:
        #     break
    
    config_save = {}
    config_save['results'] = acc_list
    try:
        with open(intermidiate_savepath, 'r') as file:
            # Try loading existing JSON data
            data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, initialize an empty list
        data = []
    except json.decoder.JSONDecodeError:
        # If the file content is not valid JSON, initialize an empty list
        data = []

    # Append the new dictionary to the data list
    data.append(config_save)

    # Open the file and overwrite the content
    with open(intermidiate_savepath, 'w') as file:
        json.dump(data, file, indent=4)

    return best_accuracy


if __name__ == "__main__":
    seeds = [0, 1, 2]  # Define your seeds here
    for seed in seeds:
        best_accuracy = main(seed)
        print(f"Seed {seed} - Best Accuracy: {best_accuracy * 100:.2f}%")