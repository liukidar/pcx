import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx
from optax._src import base 
from optax._src import combine
from optax._src import transform
import optax
from typing import Any, Callable, Optional, Union

import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
import random
import uuid
import os
import argparse

import tempfile, shutil, os, subprocess, warnings

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from pytorch_fid.fid_score_mnist import save_stats_mnist, fid_mnist
from inception_score import get_mnist_inception_score


class Model(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array],
        input_var = 1.0
    ) -> None:
        super().__init__()

        def se_energy_input(vode, rkg: px.RandomKeyGenerator = px.RKG):
            """Squared error energy function derived from a Gaussian distribution."""
            e = vode.get("h") - vode.get("u")
            return 0.5 * (e * e)/input_var


        self.act_fn = px.static(act_fn)
        
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim)]

        self.vodes = [
            pxc.Vode((hidden_dim,)) for _ in range(nm_layers-1)
        ] + [pxc.Vode((output_dim,), se_energy_input)]
        
        self.out_vode_act_fn = px.static(jax.nn.tanh)

        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(l(self.act_fn(x)))

        x = self.vodes[-1](self.out_vode_act_fn(self.layers[-1](self.act_fn(x))))

        if y is not None:
            self.vodes[-1].set("h", y)
        return self.vodes[-1].get("u")
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int,
    x: jax.Array,
    y: jax.Array,
    *,
    model: Model,
    optim_w: pxu.Optim,
    optim_h: pxu.Optim
):
    def h_step(i, x, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"], True)
        return x, None

    # print("Training!")
    model.train()
    
    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    
    # Inference steps
    pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"])


def train(dl, T,*, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim, verbose:bool = False):
    model.vodes[-1].h.frozen = True
    dl = tqdm(dl) if verbose else dl
    for x, y in dl:
        train_on_batch(T, x, y, model=model, optim_w=optim_w, optim_h=optim_h)

@pxf.jit(static_argnums=0)
def eval_on_batch(
    T: int,
    x: jax.Array, 
    *, 
    model: Model,
    optim_h: pxu.Optim
    ):
    def h_step(i, x, *, model, optim_h):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        optim_h.step(model, g["model"], True)
        return x, None

    # print("Evaluation!")  
    model.train()

    if model.vodes[-1].h.frozen:
        print("vode[-1] should not be frozen! set frozen=False before calling eval function.")

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
    
    # Inference steps
    x, y_ = pxf.scan(h_step, xs=jax.numpy.arange(T))(x, model=model, optim_h=optim_h)


def gen_imgs(dl, T, *, model: Model, optim_h: pxu.Optim):
    model.vodes[-1].h.frozen = False
    ys_ = []
    for x, y in dl:
        eval_on_batch(T, x, model=model, optim_h=optim_h)
        u = forward(x, None, model=model)
        ys_.append(u)
    return np.concatenate(ys_, axis=0) 

def tmp_save_imgs(imgs):
    tf = tempfile.NamedTemporaryFile()
    new_folder = False
    while not new_folder:
        try:
            new_folder=True
            os.makedirs("./data"+tf.name+"_")
        except OSError:
            print("ERROR")
            tf = tempfile.NamedTemporaryFile()
            new_folder=False
    for img_idx in range(len(imgs)):
        save_image(imgs[img_idx], "./data"+tf.name+"_"+"/"+str(img_idx)+".png")
    return "./data"+tf.name+"_"


def make_compressed_MNIST_files(test_dataset, data_folder):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # save test images
    test_img_folder = data_folder + "/mnist_test"
    data, label = list(test_loader)[0]
    images = data.view(-1,28,28)
    images = images/2 + 0.5     # remove normalisation
    os.makedirs(test_img_folder, exist_ok=True)
    for img_idx in tqdm(range(len(images))):
        save_image(images[img_idx], test_img_folder+"/"+str(img_idx)+".png")
    # get and save summary statistics of test images
    compressed_filename = test_img_folder + ".npz"    
    save_stats_mnist(test_img_folder, compressed_filename)    

# MCPC evaluation loop for 1D data
def eval(dl, dataset, T, *, model: Model, optim_h: pxu.Optim):
    model.vodes[-1].h.frozen = False

    # check if summary statistics of test dataset used for FID exist
    data_folder = './data'
    if not os.path.exists(data_folder + "/mnist_test.npz") :
        print(data_folder + "/mnist_test" + "does not exist")
        print("Creating compressed MNIST files for faster FID measure ...")
        make_compressed_MNIST_files(dataset, data_folder=data_folder)
 

    # generate images from model
    imgs = gen_imgs(dl, T, model=model, optim_h=optim_h)
    imgs = imgs/2 + 0.5     # from -1 -> 1 to 0 -> 1
    imgs = np.clip(imgs, 0, 1)

    # # save images
    img_folder = tmp_save_imgs(torch.tensor(imgs).reshape(-1,28,28))
    # get inceptions score
    is_mean, is_std = get_mnist_inception_score(img_folder)

    # get mnist fid
    fid = fid_mnist(data_folder + "/mnist_test.npz", img_folder, device=torch.device("cpu"), num_workers=0, verbose=False)

    shutil.rmtree(img_folder)

    return is_mean, fid, imgs

## define noisy sgd optimiser for MCPC
def sgdld(
    learning_rate: base.ScalarOrSchedule,
    momentum: Optional[float] = None,
    h_var: float = 1.0,
    gamma: float = 0.,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    seed: int = 0
    ) -> base.GradientTransformation:
    eta = 2*h_var*(1-momentum)/learning_rate if momentum is not None else 2*h_var/learning_rate
    return combine.chain(
        transform.add_noise(eta, gamma, seed),
        (transform.trace(decay=momentum, nesterov=nesterov,
                        accumulator_dtype=accumulator_dtype)
        if momentum is not None else base.identity()),
        transform.scale_by_learning_rate(learning_rate)
)



def main(args):    
    if args.is_wandb:
        wandb.init(entity="oliviers-gaspard", project="pcax")
        for key, value in wandb.config.items():
            setattr(args, key, value)
        wandb.config.update(args)
    is_wandb = args.is_wandb
    verbose = args.is_verbose
    
    px.RKG.seed(0)
    torch.manual_seed(0)
    
    batch_size = args.batch_size
    lr = args.lr_h
    momentum = args.momentum
    h_var = args.h_var
    gamma = args.gamma
    lr_p = args.lr_p
    weight_decay = args.decay_p
    input_var = args.input_var
    activation = args.activation
    latent_dim = 2
    hidden_dim = 256

    if activation == "relu":
        activation = jax.nn.relu
    elif activation == 'tanh':
        activation = jax.nn.tanh
    elif activation =='silu':
        activation = jax.nn.silu
    elif activation =='l-relu':
        activation = jax.nn.leaky_relu
    elif activation == "h-tanh":
        activation = jax.nn.hard_tanh
    else:
        raise NotImplementedError


    # Define the transformation to scale pixels to the range [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),            # Convert the image to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor to the range [-1, 1]
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_dl = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=True)
    data, label = list(train_dl)[0]
    nm_elements = len(data)
    X = (label.numpy() % 2)[:batch_size * (nm_elements // batch_size)]
    X = jax.nn.one_hot(X, 2)
    y = data.numpy()[:batch_size * (nm_elements // batch_size)]
    nm_elements_test =  1024
    X_test = np.zeros((batch_size * (nm_elements_test // batch_size), latent_dim))
    X_test[:nm_elements_test//2, 0] = 1
    X_test[nm_elements_test//2:, 1] = 1
    y_test = np.zeros((batch_size * (nm_elements_test // batch_size), 784)) # is not usedtrain_dl = list(zip(X.reshape(-1, batch_size, latent_dim), y.reshape(-1, batch_size, 784)))
    train_dl = list(zip(X.reshape(-1, batch_size, latent_dim), y.reshape(-1, batch_size, 784)))
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, latent_dim), y_test.reshape(-1, batch_size, 784)))
    
    model = Model(
        input_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=784,
        nm_layers=4,
        act_fn=activation,
        input_var = input_var
    )


    h_optimiser_fn = sgdld
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jax.numpy.zeros((batch_size, latent_dim)), None, model=model)
        optim_h = pxu.Optim(h_optimiser_fn(lr, momentum, h_var, gamma), pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True))(model))
        optim_w = pxu.Optim(optax.adamw(lr_p, weight_decay = weight_decay), pxu.Mask(pxnn.LayerParam)(model))
        # make optimiser that also optimises the activity of the model layer[-1]
        model.vodes[-1].h.frozen = False
        forward(jax.numpy.zeros((batch_size, latent_dim)), None, model=model)
        optim_h_eval = pxu.Optim(h_optimiser_fn(lr, momentum, h_var, gamma), pxu.Mask(pxu.m(pxc.VodeParam))(model))
        model.vodes[-1].h.frozen = True
    
    nm_epochs = 100
    T = 250
    T_eval = 10000

    best_is = 0
    best_imgs = None
    for e in range(nm_epochs):
        random.shuffle(train_dl)
        train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h, verbose=verbose)
        if e % 10 == 9:
            is_, fid, imgs = eval(test_dl, test_dataset, T_eval, model=model, optim_h=optim_h_eval)
            if verbose:
                print(f"Epoch {e + 1}/{nm_epochs} - Inception score: {is_ :.2f}, FID score: {fid :.2f}")
            if is_wandb:
                wandb.log({"is":is_, "fid":fid})
            if is_ > best_is:
                best_imgs = imgs
                best_is = is_
    is_, fid, imgs = eval(test_dl, test_dataset, T_eval, model=model, optim_h=optim_h_eval)
    if verbose:
        print(f"Epoch {e+1}/{nm_epochs} - Inception score: {is_ :.2f}, FID score: {fid :.2f}")
    if is_wandb:
        wandb.log({"is":is_, "fid":fid})
    if is_ > best_is:
        best_imgs = imgs
        best_is = is_
    

    images_reshaped = best_imgs.reshape(-1, 28, 28)
    fig, axes = plt.subplots(10, 10, figsize=(10,10))
    axes = axes.ravel()

    for i in np.arange(0, 50):
        axes[i].imshow(images_reshaped[i], cmap='gray')
        axes[i].axis('off')

    for i in np.arange(0, 50):
        axes[i+50].imshow(images_reshaped[-50 + i], cmap='gray')
        axes[i+50].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plot_filename = f"plot_{uuid.uuid4().hex}.png"
    plt.savefig(plot_filename)
    if is_wandb:
        wandb.log({"plot": wandb.Image(plot_filename)})
        os.remove(plot_filename)
    
    if verbose:
       plt.show() 
    
    if is_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training a MCPC model on the iris dataset.',
        fromfile_prefix_chars='@'
    )

    parser.add_argument('--batch-size', type=int, default=150, choices=[150, 300, 600, 900],
        help='batch size')
    parser.add_argument('--lr-h', type=float, default=0.01, 
        help='learning rate of neurons')
    parser.add_argument('--momentum', type=float, default=None,
        help='momentum of neurons')
    parser.add_argument('--h-var', type=float, default=1.0,
        help='variance of layer based on noise')
    parser.add_argument('--gamma', type=float, default=0.0,
        help='decay exponent of activity noise')    
    parser.add_argument('--lr-p', type=float, default=0.001,
        help='learning rate of parameters')
    parser.add_argument('--decay-p', type=float, default=0.0,
        help='decay of parameters')
    parser.add_argument('--input-var', type=float, default=1.0,
        help='variance of input layers from energy')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'silu'],
        help='activation function')    
    parser.add_argument('--is-wandb', type=lambda x: (str(x).lower() == 'true'), default=False,
        help='activation function')    
    parser.add_argument('--is-verbose', type=lambda x: (str(x).lower() == 'true'), default=True,
        help='print outputs')    
    
    args = parser.parse_args()

    print(args)
    main(args)
