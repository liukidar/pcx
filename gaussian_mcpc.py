# Tutorial #6: Monte Carlo Predictive Coding

# In this notebook we will see how to create and train a simple MCPC model to learn a Gaussian data distribution.
from typing import Callable

# These are the default import names used in tutorials and documentation.
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx

import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu

# px.RKG is the default key generator used in pcax, which is used as default
# source of randomness within pcax. Here we set its seed to 0 for more reproducibility.
# By default it is initialised with the system time.
px.RKG.seed(0)
# We create our model, which inherits from pxc.EnergyModule, so to have access to the notion
# energy. The constructor takes in input all the hyperparameters of the model. Being static
# values, if we intend to save them withing the model we must wrap them into a 'StaticParam'.
class Model(pxc.EnergyModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nm_layers: int,
        act_fn: Callable[[jax.Array], jax.Array]
    ) -> None:
        super().__init__()

        self.act_fn = px.static(act_fn)
        
        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim)]

        # the default ruleset for a Vode is: `{"STATUS.INIT": ("h, u <- u",),}` which means:
        # "if the status is set to 'STATUS.INIT', everytime I set 'u', save that value not only
        # in 'u', but also in 'x', which is exactly the behvaiour of a forward pass.
        # By default if not specified, the behaviour is '* <- *', i.e., save everything passed
        # to the vode via __call__ (remember vode(a) equals to vode.set("u", a)).
        #
        # Since we are doing classification, we replace the last energy with the equivalent of
        # cross entropy loss for predictive coding.
        self.vodes = [
            pxc.Vode((hidden_dim,)) for _ in range(nm_layers - 1)
        ] + [pxc.Vode((output_dim,), pxc.ce_energy)]
        
        # 'frozen' is not a magic word, we define it here and use it later to distinguish between
        # vodes we want to differentiate or not.
        # NOTE: any attribute of a Param (except its value) is treated automatically as static,
        # no need to specify it (but it's possible if you like more consistency,
        # i.e., ...frozen = px.static(True)).
        self.vodes[-1].h.frozen = True

    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            # remember 'x = v(a)' corresponds to v.set("u", a); x = v.get("x")
            #
            # note that 'self.act_fn' is a StaticParam, so to access it we would have to do
            # self.act_fn.get()(...), however, all standard methods such as __call__ and
            # __getitem__ are overloaded such that 'self.act_fn.__***__' becomes
            # 'self.act_fn.get().__***__'
            x = v(self.act_fn(l(x)))

        x = self.vodes[-1](self.layers[-1](x))

        if y is not None:
            # if the target label is provided (e.g., during training), we save it to the last
            # vode. Given that the 'froze' it, its value will not be upadated during inference,
            # so we need to fix it only once for each new sample, usually during the init step.
            self.vodes[-1].set("h", y)

        # at least with this architecture, the input activation of the last vode is the actual
        # output of the model ('h' is fixed to the label during training or 'h = u' during eval)
        return self.vodes[-1].get("u")
# vmap is used to specify the batch dimension of the input data. Remember jax doesn't handle it
# implicitly but relies on the user to explicitly tell it over which dimension to parallelise the
# computation. That is, we always define a computational graph on a single sample, and then batch
# the computation over the given mini-batch. We use the jax syntax for in_axes, out_axes, axis_name,
# and the introduce a new parameter, kwargs_mask, to specify the batch information over the kwargs
# (which, just as a reminder, have the property of being automatically tracked by pcax).
# pxu.utils.mask has an in-depth explanation about how masking work. Here, we simply use the Mask
# object, which, in this case, replaces every parameter that matches any of the given types with '0',
# meaning that their value is batched over the 0th dimension (which is the case for the vode values
# and caches), and with 'None' the non matching ones (such as the weights, which are shared across
# different samples).
# Both positional input arguments and output are batched over the 0th dimension, so we specify it.
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)

# Similarly here, we specify 'out_axes=(None, 0)' since the function returns two values, the first
# a single float storing the total energy of the model (not batched, but summed over the batch
# dimension; this is a requirement of the gradient transformation, which jax requires taking a
# scalar function in input and so a single scalar output). To follow on this, 'axis_name' is specified
# so that we can return the sum over the batch dimension as required (this is standard jax syntax).
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.psum(model.energy().sum(), "batch"), y_
# JIT is Just In Time compilation, which effectively compiles our code for fast CPU/GPU executioning
# removing all python overhead.
# 'T' is an hyperparameter that determines the number of inferences steps (and therefore the computational flow).
# A such, it must be a static value. We can either specify it using 'static_argnums' (which however is only available
# when using 'jit'), or pass it as a static parameter, in which case we would to 'train_on_batch(px.static(T), ...)'.
#
# Remember that pcax distinguishes between positional and keyword arguments, tracking only the parameters in latter ones.
# Since we don't care about tracking of 'x' and 'y', we pass them as simple jax.Arrays as positional arguments. On the
# other hand, both the model and the optimizers, may have parameters that are going to change and we want to track, so
# we pass them as keyword arguments.
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
    print("Training!")  # this will come in handy later

    # This only sets an internal flag to be "train" (instead of "eval")
    model.train()
    
    # 'pxu.step' is an utility function that does two things:
    # - sets the status to the provided one (default is 'None')
    #   (and resets it to 'None' afterwards);
    # - clears the target parameters if clear_params is specified
    # (normally we want to clear the vode cache, such as activation and energy,
    # after each step).
    #
    # pxc.STATUS.INIT triggers the only default vode ruleset defined, as
    # previously explained.

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)
    
    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            # 'm' is a masking object with a couple of useful methods to create complex masking functions.
            # Here, we use it to target all VodeParams that are not forzen (again, frozen is a totally custom
            # attribute we, as users, we decided to use above in the model and here).
            #
            # As jax expects, we distinguish between Parameters to differentiate ('True') and the rest ('False')
            #
            # 'e', 'y_' are the values returned by the 'energy' function defined above
            (e, y_), g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        
        # the returned gradient has the same structure of the function input. In this case, since we didn't use
        # 'argnums' (jax argument of 'value_and_grad'), we only return the gradient with respect to the keyword
        # arguments, that can be accessed as a dictionary. If we also had positional arguments gradients, we
        # would have 'g = (positional_grad, keyword_grad)', so that, for example, the gradient of 'model' would
        # be at 'g[1]["model"]'.
        optim_h.step(model, g["model"])

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
    optim_w.step(model, g["model"])
import numpy as np
from scipy.stats import wasserstein_distance

@pxf.jit()
def eval_on_batch(
    T: int,
    x: jax.Array, 
    *, 
    model: Model,
    optim_h: pxu.Optim
    ):
    print("Evaluation!")  
    model.train()

    if model.vodes[-1].h.frozen:
        print("vode[-1] should not be frozen! set frozen=False before calling eval function.")

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, None, model=model)
    
    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (e, y_), g = pxf.value_and_grad(
                pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                has_aux=True
            )(energy)(x, model=model)
        
        optim_h.step(model, g["model"])

    # Learning step
    return y_

from tqdm import tqdm
# Standard training loop
def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.vodes[-1].h.frozen = True
    for x, y in tqdm(dl):
        train_on_batch(T, x, jax.nn.one_hot(y, 2), model=model, optim_w=optim_w, optim_h=optim_h)

# Standard evaluation loop
def eval(dl, T, *, model: Model, optim_h: pxu.Optim):
    model.vodes[-1].h.frozen = False
    ys = []
    ys_ = []
    
    for x, y in dl:
        y_ = eval_on_batch(T, x, model=model, optim_h=optim_h)
        ys.append(y)
        ys_.append(y_)

    ys = np.concatenate(ys, axis=0)
    ys_ = np.concatenate(ys_, axis=0)

    return wasserstein_distance(ys, ys_), ys_
import optax

batch_size = 32

model = Model(
    input_dim=1,
    hidden_dim=1,
    output_dim=1,
    nm_layers=2,
    act_fn= lambda x:x
)

# initialise model in consistent way
# model.layers[0].nn.weight.set([[0.0]])
# model.layers[0].nn.bias.set([1.0])
# model.layers[1].nn.weight.set([[1.0]])
# model.layers[1].nn.bias.set([0.0]);
h_optimiser_fn = optax.noisy_sgd
lr = 1e-2
eta = 2/lr
gamma = 0
lr_p = 1e-2

# only thing to note here is how optimizers are created. In particular,
# we first want all the parameters of the model to exist, so that the optimizers
# can capture them for optimization. This requires performing a dummy forward pass.
# Note that the batch_size is an hyperparameter of the model and determines, among
# other things, the shape of the Vode parameters, and thus must be kept as much
# constant as possible (each change would trigger ricompilation of the jitted functions).
with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(jax.numpy.zeros((batch_size, 1)), None, model=model)
    
    # 'pxu.Optim' accepts a optax optimizer and the parameters pytree in input. pxu.Mask
    # can be used to partition between target parameters and not: when no 'map_to' is
    # provided, such as here, it acts as 'eqx.partition', using pxc.VodeParam as filter.
    optim_h = pxu.Optim(h_optimiser_fn(lr, eta, gamma), pxu.Mask(pxc.VodeParam)(model))
    optim_w = pxu.Optim(optax.adam(lr_p), pxu.Mask(pxnn.LayerParam)(model))
import matplotlib.pyplot as plt
# generate Gaussian data

mean = 1
var = 2

nm_elements = 1024
X = np.zeros((batch_size * (nm_elements // batch_size), 1))
y = np.random.randn(batch_size * (nm_elements // batch_size)).reshape(-1,1) * np.sqrt(var) + mean

nm_elements_test = 1024
X_test = np.zeros((batch_size * (nm_elements_test // batch_size), 1))
y_test = np.random.randn(batch_size * (nm_elements // batch_size)).reshape(-1,1) * np.sqrt(var) + mean

plt.hist(y, alpha = 0.5, density=True)
plt.hist(y_test, alpha = 0.5, density=True);
# from sklearn.datasets import make_moons

# # this is unrelated to pcax: we generate and display the training set.
# nm_elements = 1024
# X, y = make_moons(n_samples=batch_size * (nm_elements // batch_size), noise=0.2, random_state=42)

# # Plot the dataset
# plt.figure(figsize=(6, 4))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
# plt.title("Two Moons Dataset")
# plt.show()
# we split the dataset in training batches and do the same for the generated test set.
train_dl = list(zip(X.reshape(-1, batch_size, 1), y.reshape(-1, batch_size, 1)))
test_dl = tuple(zip(X_test.reshape(-1, batch_size, 1), y_test.reshape(-1, batch_size, 1)))
import random

nm_epochs = 256 // (nm_elements // batch_size)

# Note how the text "Training!" appears only once. This is because 'train_on_batch' is executed only once,
# and then its compiled equivalent is instead used (which only cares about what happens to jax.Arrays and
# discards all python code).
T = 150
T_eval = 1000
for e in range(nm_epochs):
    random.shuffle(train_dl)
    train(train_dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
    # w, y_pred = eval(test_dl, T_eval, model=model, optim_h=optim_h)
    
    # # We print the average shift of the first vode during the inference steps. Note that it does not depend on
    # # the choice for the batch_size (feel free to play around with it, remember to reset the notebook if you
    # # you change it). This is because we multiply the learning rate of 'optim_h' by the batch_size. This is 
    # # because the total energy is averaged over the batch dimension (as required for the weight updates),
    # # so we need to scale the learning rate accordingly for the vode updates.
    # print(f"Epoch {e + 1}/{nm_epochs} - Wasserstein distance: {w :.2f}%")
plt.hist(y, label = "data")
plt.hist(y_pred, lable = "generated")
plt.ylabel("pdf")
plt.xlabel("y")
plt.show()