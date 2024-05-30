import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx
from optax._src import base 
from optax._src import combine
from optax._src import transform
import optax
from typing import Any, Callable, Optional, Union
import ot

import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import wandb
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from scipy.stats import gaussian_kde
import random
import uuid
import os
import argparse

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
        
        
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes, self.layers):
            x = v(l(self.act_fn(x)))

        if y is not None:
            self.vodes[-1].set("h", y)
        return self.vodes[-1].get("u")
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.psum(model.energy().sum(), "batch"), y_
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
        optim_h.step(model, g["model"])
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
        optim_h.step(model, g["model"])
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

# MCPC evaluation loop for 1D data
def eval(dl, T, *, model: Model, optim_h: pxu.Optim):
    model.vodes[-1].h.frozen = False
    ys = []
    ys_ = []
    
    for x, y in dl:
        eval_on_batch(T, x, model=model, optim_h=optim_h)
        ys.append(y)
        ys_.append(model.vodes[-1].get("h"))

    ys = np.concatenate(ys, axis=0)
    ys_ = np.concatenate(ys_, axis=0)

    a = np.ones(len(ys)) / len(ys)
    b = np.ones(len(ys_)) / len(ys_)
    M = ot.dist(ys, ys_)
    emd_distance = ot.emd2(a, b, M)
    return emd_distance, ys_


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
    
    batch_size = args.batch_size
    lr = args.lr_h
    momentum = args.momentum
    h_var = args.h_var
    gamma = args.gamma
    lr_p = args.lr_p
    weight_decay = args.decay_p
    input_var = args.input_var
    activation = args.activation

    if activation == "relu":
        activation = jax.nn.relu
    elif activation == 'tanh':
        activation = jax.nn.tanh
    elif activation =='silu':
        activation = jax.nn.silu
    else:
        raise NotImplementedError


    # get data
    iris = load_iris()
    y = iris.data
    y_data = y[:,(0,2)]
    y_data = (y_data - y_data.mean(0)) / y_data.std(0)
    n = len(y_data)

    nm_elements = batch_size*n
    X = np.zeros((batch_size * (nm_elements // batch_size), 2))
    y = np.repeat(y_data, batch_size, axis=0)
    np.random.shuffle(y)

    if n % batch_size == 0:
        nm_elements_test = n
    elif batch_size % n == 0:
        nm_elements_test = batch_size
    else:
        raise NotImplementedError
    X_test = np.zeros((batch_size * (nm_elements_test // batch_size), 2))
    y_test = np.repeat(y_data, nm_elements_test // n, axis=0)


    nm_elements_gen = n * 30  # 150 x 30
    X_gen = np.zeros((batch_size * (nm_elements_gen // batch_size), 2))
    y_gen = np.repeat(y_data, nm_elements_gen // n, axis=0)

    # we split the dataset in training batches and do the same for the generated test set.
    train_dl = list(zip(X.reshape(-1, batch_size, 2), y.reshape(-1, batch_size, 2)))
    test_dl = tuple(zip(X_test.reshape(-1, batch_size, 2), y_test.reshape(-1, batch_size, 2)))
    gen_dl = tuple(zip(X_gen.reshape(-1, batch_size, 2), y_gen.reshape(-1, batch_size, 2)))
    

    model_nn = Model(
        input_dim=2,
        hidden_dim=64,
        output_dim=2,
        nm_layers=3,
        act_fn=activation,
        input_var = input_var
    )


    h_optimiser_fn = sgdld
    with pxu.step(model_nn, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jax.numpy.zeros((batch_size, 2)), None, model=model_nn)
        optim_h = pxu.Optim(h_optimiser_fn(lr, momentum, h_var, gamma), pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True))(model_nn))
        optim_w = pxu.Optim(optax.adamw(lr_p, weight_decay=weight_decay), pxu.Mask(pxnn.LayerParam)(model_nn))
        # make optimiser that also optimises the activity of the model layer[-1]
        model_nn.vodes[-1].h.frozen = False
        forward(jax.numpy.zeros((batch_size, 2)), None, model=model_nn)
        optim_h_eval = pxu.Optim(h_optimiser_fn(lr, momentum, h_var, gamma), pxu.Mask(pxu.m(pxc.VodeParam))(model_nn))
        model_nn.vodes[-1].h.frozen = True
    import random

    nm_epochs = 2*5120 // (nm_elements // batch_size)

    T = 250
    T_eval = 10000
    w, y_ = eval(test_dl, T = T_eval, model=model_nn, optim_h=optim_h_eval)
    if verbose:
        print(f"Epoch {0}/{nm_epochs} - Wasserstein distance: {w :.2f}")
    if is_wandb:
        wandb.log({"dist":w})
    for e in range(nm_epochs):
        random.shuffle(train_dl)
        train(train_dl, T=T, model=model_nn, optim_w=optim_w, optim_h=optim_h, verbose=verbose)
        if e %5 == 4:
            w, y_ = eval(test_dl, T = T_eval, model=model_nn, optim_h=optim_h_eval)
            if verbose:
                print(f"Epoch {e + 1}/{nm_epochs} - Wasserstein distance: {w :.2f}")
            if is_wandb:
                wandb.log({"dist":w})
    
    w, y_ = eval(gen_dl, T = T_eval, model=model_nn, optim_h=optim_h_eval)
    if verbose:
        print(f"Epoch {e + 1}/{nm_epochs} - Wasserstein distance: {w :.2f}")
    if is_wandb:
        wandb.log({"dist":w})
    
    # print(f"Learned data distribution has mean {y_.mean():.2f} and var {y_.var():.2f} ")
    if verbose:
        print(f"Learned parameters weight {model_nn.layers[-1].nn.weight.get()} and bias {model_nn.layers[0].nn.bias.get()}")
    
    # Create a 2D Gaussian kernel density estimate
    data = np.vstack(y_.transpose())
    kde = gaussian_kde(data)
    # Define grid points
    x_grid = np.linspace(data[0].min(), data[0].max(), 100)
    y_grid = np.linspace(data[1].min(), data[1].max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Plot contour lines
    plt.contour(X, Y, Z, levels=10, cmap='Blues')
    plt.scatter(y_test[:,0], y_test[:,1], marker="x", color="r", label="data")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend()
    plt.tight_layout()
    plot_filename = f"plot_{uuid.uuid4().hex}.svg"
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
