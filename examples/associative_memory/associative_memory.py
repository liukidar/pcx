from typing import Callable

# Core dependencies
import jax
import jax.numpy as jnp

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf

import optax
import torch
import numpy as np
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset

# utility libraries
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import argparse
import urllib.request
import zipfile
from PIL import Image
import json

STATUS_FORWARD = "forward"


class Decoder(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)

        self.layers = (
            [pxnn.Linear(input_dim, hidden_dim)]
            + [pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)]
            + [pxnn.Linear(hidden_dim, output_dim)]
        )

        # We initialise the first node to zero.
        # We use 'zero_energy' as we do not want any prior on the first layer.
        self.vodes = (
            [
                pxc.Vode(
                    (input_dim,),
                    energy_fn=pxc.zero_energy,
                    ruleset={pxc.STATUS.INIT: ("h, u <- u:to_zero",)},
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros(n.shape)},
                )
            ]
            + [
                # we stick with default forward initialisation for now for the remaining nodes,
                # however we enable a "forward mode" where we forward the incoming activation instead
                # of the node state; this is used during evaluation to generate the encoded output.
                pxc.Vode(
                    (hidden_dim,),
                    ruleset={
                        pxc.STATUS.INIT: ("h, u <- u:to_zero",),
                        STATUS_FORWARD: ("h -> u",)
                    },
                    tforms={"to_zero": lambda n, k, v, rkg: jnp.zeros_like(v)},
                )
                for _ in range(nm_layers - 1)
            ]
            + [pxc.Vode((output_dim,))]
        )
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array | None, y: jax.Array | None):
        # The defined ruleset for the first node is to set the hidden state to zero,
        # independent of the input, so we always pass '-1'.

        # note that here since we have a decoding pc, y is the image; x is simply a pseudo placeholder
        x = self.vodes[0](-1)
        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i != len(self.layers) - 1 else lambda x: x
            x = act_fn(layer(x))
            x = self.vodes[i + 1](x)

        if y is not None:
            self.vodes[-1].set("h", y)

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

class TinyImageNet(Dataset):
    def __init__(self, root_dir='../data', transform=None, download=False, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # Check if dataset exists, if not, download it
        if download and not os.path.exists(os.path.join(self.root_dir, 'tiny-imagenet-200')):
            self.download()

        self.train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
        # Create a mapping from string class IDs to integer labels
        self.class_to_label = {class_name: class_id for class_id, class_name in enumerate(os.listdir(self.train_dir))}

        if self.train:
            self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
            self.image_paths = []
            self.labels = []
    
            for class_id, class_name in enumerate(os.listdir(self.data_dir)):
                class_dir = os.path.join(self.data_dir, class_name, 'images')
                for image_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(class_id)
        else:
            self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'images')
            with open(os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                self.image_paths = [os.path.join(self.data_dir, line.split('\t')[0]) for line in lines]
                # Use the mapping to convert string class IDs to integer labels
                self.labels = [self.class_to_label[line.split('\t')[1]] for line in lines]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def download(self):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        filename = os.path.join(self.root_dir, "tiny-imagenet-200.zip")

        # Download the dataset
        urllib.request.urlretrieve(url, filename)

        # Extract the dataset
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)

        # Remove the downloaded zip file after extraction
        os.remove(filename)

def get_dataloader(batch_size: int, sample_size: int, train, seed=None):
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            lambda x: x.numpy(),
        ]
    )

    dataset = TinyImageNet(
        "../../../datasets",
        transform=t,
        download=True,
        train=train,
    )

    # Generate random indices for the subset
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:sample_size]
    subset = Subset(dataset, indices)

    dataloader = TorchDataloader(
        subset,
        batch_size=batch_size,
        shuffle=(seed is None),
        num_workers=4,
    )

    return dataloader


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Decoder):
    return model(x, y)


@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Decoder):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, 
    x: jax.Array, # image
    y: jax.Array, # label
    *, 
    model: Decoder, 
    optim_w: pxu.Optim, 
    optim_h: pxu.Optim
):
    model.train()

    def h_step(i, x, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"], scale_by_batch_size=False)
        return x, None

    learning_step = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(y, x, model=model)

    pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = learning_step(x, model=model)
    optim_w.step(model, g["model"])


@pxf.jit(static_argnums=0)
def denoise_on_batch(
    T: int, 
    x: jax.Array, 
    x_train: jax.Array,
    y: jax.Array, 
    *, 
    model: Decoder, 
    optim_h: pxu.Optim,
):
    model.eval()

    """Note that frozen isn't any pre-defined variable at all. In model we set bottom layer's h.frozen=True
    Here, we ask Jax to only to differentiate those variables with frozen=False (i.e., has not frozen)
    Therefore, to get a denoising/completion effect we can set fronzen=False in a `decorrupt_on_batch` function.
    """

    def h_step(i, x, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"], scale_by_batch_size=False)
        return x, None

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # forward will set vodes[-1] to the noisy image
        forward(y, x, model=model)
    
    # for memory, optimize the sensory level values nodes too
    model.vodes[-1].h.frozen = False
    pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = model.vodes[-1].get("h")

    x_hat = jnp.clip(x_hat, 0.0, 1.0)
    l = jnp.square(x_hat.flatten() - x_train.flatten()).mean()

    return l, x_hat

@pxf.jit(static_argnums=0)
def unmask_on_batch(
    T: int, 
    x: jax.Array, 
    x_train: jax.Array,
    y: jax.Array, 
    *, 
    model: Decoder, 
    optim_h: pxu.Optim,
):
    model.eval()

    """Note that frozen isn't any pre-defined variable at all. In model we set bottom layer's h.frozen=True
    Here, we ask Jax to only to differentiate those variables with frozen=False (i.e., has not frozen)
    Therefore, to get a denoising/completion effect we can set fronzen=False in a `decorrupt_on_batch` function.
    """

    def h_step(i, y, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(y, model=model)
        optim_h.step(model, g["model"], scale_by_batch_size=False)

        model.vodes[-1].h.set(
            model.vodes[-1].h.reshape((-1, 3, 64, 64)).at[:, :, :32].set(
                x.reshape((-1, 3, 64, 64))[:, :, :32]
            ).reshape((-1, 3*64*64))
        )
        return x, None

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        # forward will set vodes[-1] to the masked image x
        forward(y, x, model=model)
    
    # for memory, optimize the sensory level values nodes too
    model.vodes[-1].h.frozen = False
    pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)

    with pxu.step(model, STATUS_FORWARD, clear_params=pxc.VodeParam.Cache):
        x_hat = model.vodes[-1].get("h")

    x_hat = jnp.clip(x_hat, 0.0, 1.0)
    l = jnp.square(x_hat.flatten() - x_train.flatten()).mean()

    return l, x_hat

def train(dl, T, *, model: Decoder, optim_w: pxu.Optim, optim_h: pxu.Optim, verbose=True):
    if verbose:
        dl = tqdm(dl)
    for x, y in dl:
        train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)


def eval(dl, T, *, model: Decoder, optim_h: pxu.Optim, corruption: str = "noise"):
    losses = []

    for x, y in dl:
        if corruption == "noise":
            x_c = x + jax.random.normal(px.RKG(), x.shape) * 0.2
            e, x_hat = denoise_on_batch(T, x_c, x, y, model=model, optim_h=optim_h)
        elif corruption == "mask":
            x_c = x.reshape((-1, 3, 64, 64)).copy()
            x_c[:, :, 32:] = 0
            x_c = x_c.reshape((-1, 3*64*64))
            e, x_hat = unmask_on_batch(T, x_c, x, y, model=model, optim_h=optim_h)
        losses.append(e)

    return np.mean(losses), x, x_c, x_hat # return only the final batch of images for visualization

parser = argparse.ArgumentParser()
parser.add_argument('--corruption', type=str, default='noise', help='corruption type')
parser.add_argument('--hidden_dim', type=int, default=512, help='size of the hidden layers')
parser.add_argument('--sample_size', type=int, default=50, help='number of images to memorize')
args = parser.parse_args()

hidden_dim = args.hidden_dim
sample_size = args.sample_size
with open('hps.json', 'r') as file:
    params = json.load(file)
    hps = params[args.corruption][str(hidden_dim)][str(sample_size)]
    h_lr = hps['h_lr']
    w_lr = hps['w_lr']
    T_train = hps['T_train']
    T_gen = hps['T_gen']
print(h_lr, w_lr, T_train, T_gen)

# fixed variables
batch_size = 50
nm_epochs = 500
image_size = 64*64*3

# image recall, specify the save path as you like
save_path = f'./results/{args.corruption}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

model = Decoder(input_dim=512, hidden_dim=hidden_dim, output_dim=image_size, nm_layers=3, act_fn=jax.nn.tanh)
with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(None, jnp.zeros((batch_size, image_size)), model=model)

    optim_h = pxu.Optim(optax.sgd(h_lr), pxu.Mask(pxc.VodeParam)(model))
    optim_w = pxu.Optim(optax.adamw(w_lr), pxu.Mask(pxnn.LayerParam)(model))

train_dataloader = get_dataloader(batch_size, sample_size, train=True)

energies = []
for i in range(nm_epochs):
    train(train_dataloader, T=T_train, model=model, optim_w=optim_w, optim_h=optim_h, verbose=True)
    e = energy(None, model=model)[0]
    energies.append(e)
    print(f"Epoch {i + 1}/{nm_epochs} - Energy: {e:.4f}")

recon_mse, x, x_c, x_hat = eval(train_dataloader, T=T_gen, model=model, optim_h=optim_h, corruption=args.corruption)

# visualizations
print(recon_mse)
n_show = 5
fig, ax = plt.subplots(3, 10, figsize=(10, 3))
for i in range(10):
    ax[0, i].imshow(x[i].reshape((3, 64, 64)).transpose(1, 2, 0))
    ax[0, i].axis('off')
    ax[1, i].imshow(x_c[i].reshape((3, 64, 64)).transpose(1, 2, 0))
    ax[1, i].axis('off')
    ax[2, i].imshow(x_hat[i].reshape((3, 64, 64)).transpose(1, 2, 0))
    ax[2, i].axis('off')

for a in ax.flatten():
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_aspect("auto")
plt.subplots_adjust(wspace=0., hspace=0.1)
plt.savefig(os.path.join(save_path, 'reconstruction.pdf'), bbox_inches='tight')


