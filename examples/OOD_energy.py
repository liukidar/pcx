# built-in
import json
import time
import os
import argparse

# choose the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# disable preallocation of memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 3rd party
import numpy as np
import pickle

# own
from utils import create_new_test_loader_with_batch_size, get_dataloaders, set_random_seed, DatasetProcessorData

# now set the seed
set_random_seed(42)

###################################### start of model related code ##########################################

from typing import Callable

# Core dependencies
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
print(jax.default_backend())  # print backend type

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu


# Model definition
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
        self._dim_input = input_dim
        self._dim_output = output_dim
        
        self.act_fn = px.static(act_fn)

        self.layers = [pxnn.Linear(input_dim, hidden_dim)] + [
            pxnn.Linear(hidden_dim, hidden_dim) for _ in range(nm_layers - 2)
        ] + [pxnn.Linear(hidden_dim, output_dim)]
        
        self.vodes = [
            pxc.Vode((hidden_dim,)) for _ in range(nm_layers - 1)
        ] + [pxc.Vode((output_dim,), pxc.ce_energy)]

        self.vodes[-1].h.frozen = True

    def num_parameters(self):
        """
        Calculate the total number of parameters in the model.
        Args:
            model: The model object containing layers with weights and biases.
        Returns:
            int: Total number of parameters in the model.
        """
        return sum(layer.nn.weight.size + layer.nn.bias.size for layer in self.layers)

    def save(self, file_name):
        pxu.save_params(self, file_name)


    def __call__(self, x, y):
        for v, l in zip(self.vodes[:-1], self.layers[:-1]):
            x = v(self.act_fn(l(x)))
        x = self.vodes[-1](self.layers[-1](x))
        
        if y is not None:
            self.vodes[-1].set("h", y)
        return self.vodes[-1].get("u")


# Training and evaluation functions
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model: Model):
    return model(x, y)

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model: Model):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "batch"), y_

@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    model.train()

    # init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model)

    # reinitialise the optimiser state between different batches (NOTE: this is just educational and not needed here because the SGD we use is not-stateful due to lack of momentum)
    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(energy)(x, model=model)
            optim_h.step(model, g["model"], True)

    # learning (weight update) step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(x, model=model)
        optim_w.step(model, g["model"])

@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: Model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, y, model=model).argmax(axis=-1)
        e = model.vodes[-1].energy()
    
    # Convert y from one-hot encoding to class indices
    y_indices = y.argmax(axis=-1)
    
    return (y_ == y_indices).mean(), y_, e.mean()

@pxf.jit()
def predict_with_logits_on_batch(x: jax.Array, *, model: Model):
    model.eval()
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        logits_ = forward(x, None, model=model)
        y_ = logits_.argmax(axis=-1)

    return y_, logits_

def train(dl, T, *, model: Model, optim_w: pxu.Optim, optim_h: pxu.Optim):
    for x, y in dl:
        train_on_batch(T, x, jax.nn.one_hot(y, 10), model=model, optim_w=optim_w, optim_h=optim_h)

      
def eval(dl, *, model: Model):
    acc = []
    es = []
    ys_ = []
    for x, y in dl:
        a, y_, e = eval_on_batch(x, jax.nn.one_hot(y, 10), model=model)
        acc.append(a)
        es.append(e)
        ys_.append(y_)


    return float(np.mean(acc)), np.concatenate(ys_), float(np.mean(es))

def predict_with_logits(dl, *, model: Model):
    ys_ = []
    logits_ = []
    for x, _ in dl:
        y_, logits = predict_with_logits_on_batch(x, model=model)
        ys_.append(y_)
        logits_.append(logits)
    return np.concatenate(ys_), np.concatenate(logits_)


@pxf.jit(static_argnums=0)
def eval_inference_on_batch(T: int, x: jax.Array, y: jax.Array, *, model: Model, optim_h: pxu.Optim):
    model.eval()

    # init step and also compute the energy of the model with the given input and output before the inference steps (i.e. without minimising the energy)
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, model=model) # this works
    
        e_pre = model.energy()
    
    # shorthand for value and grad computation on the energy function
    inference_step = pxf.value_and_grad(pxu.Mask(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]), has_aux=True)(energy)

    # reinitialise the optimiser state between different batches (NOTE: this is just educational and not needed here because the SGD we use is not-stateful due to lack of momentum)
    optim_h.init(pxu.Mask(pxc.VodeParam)(model))

    # inference steps
    for T_i in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = inference_step(x, model=model)
            optim_h.step(model, g["model"], True)
        
    # Only compute and store energy on the last iteration
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        #e_post_luca, _ = energy(x, model=model) # this returns average energy over batch as single scalar value. This works even without supplying y. because the y value is set in the forward pass above.
        
        forward(x, y, model=model) # this works
        e_post = model.energy() # this returns energy per sample in the batch


    return e_pre, e_post

def eval_inference(dl, T, *, model: Model, optim_h: pxu.Optim):
    for x, y in dl:
        eval_inference_on_batch(T, x, jax.nn.one_hot(y, 10), model=model, optim_h=optim_h)

###################################### end of model related code ##########################################


# ID data

MNIST_dataset_name = "mnist"
train_subset_size = 30000
batch_size = 128
noise_level = 0.0 # this means that the validation set will be 20% of the train_subset_size (6000 samples)
# get the ID dataloaders
MNIST_dataset = get_dataloaders(MNIST_dataset_name, train_subset_size, batch_size, noise_level)
# Check the sizes of the datasets
print(f"Training set: {len(MNIST_dataset.train_loader.sampler)} samples")
print(f"Validation set: {len(MNIST_dataset.val_loader.sampler)} samples")
print(f"Test set: {len(MNIST_dataset.test_loader.dataset)} samples")
print()

# now load FMNIST
FMNIST_dataset_name = "fmnist"
# get the FMNIST dataloaders
FMNIST_dataset = get_dataloaders(FMNIST_dataset_name, train_subset_size, batch_size, noise_level)
# Check the sizes of the datasets
print(f"FMNIST Training set: {len(FMNIST_dataset.train_loader.sampler)} samples")
print(f"FMNIST Validation set: {len(FMNIST_dataset.val_loader.sampler)} samples")
print(f"FMNIST Test set: {len(FMNIST_dataset.test_loader.dataset)} samples")
print()

# now load EMNIST
EMNIST_dataset_name = "emnist"
# get the EMNIST dataloaders
EMNIST_dataset = get_dataloaders(EMNIST_dataset_name, train_subset_size, batch_size, noise_level)
# Check the sizes of the datasets
print(f"EMNIST Training set: {len(EMNIST_dataset.train_loader.sampler)} samples")
print(f"EMNIST Validation set: {len(EMNIST_dataset.val_loader.sampler)} samples")
print(f"EMNIST Test set: {len(EMNIST_dataset.test_loader.dataset)} samples")
print()

# now load KMNIST
KMNIST_dataset_name = "kmnist"
# get the KMNIST dataloaders
KMNIST_dataset = get_dataloaders(KMNIST_dataset_name, train_subset_size, batch_size, noise_level)
# Check the sizes of the datasets
print(f"KMNIST Training set: {len(KMNIST_dataset.train_loader.sampler)} samples")
print(f"KMNIST Validation set: {len(KMNIST_dataset.val_loader.sampler)} samples")
print(f"KMNIST Test set: {len(KMNIST_dataset.test_loader.dataset)} samples")
print()

# now load notMNIST
notMNIST_dataset_name = "notmnist"
# get the notMNIST dataloaders
notMNIST_dataset = get_dataloaders(notMNIST_dataset_name, train_subset_size, batch_size, noise_level)
# Check the sizes of the datasets
print(f"notMNIST Training set: {len(notMNIST_dataset.train_loader.sampler)} samples")
print(f"notMNIST Validation set: {len(notMNIST_dataset.val_loader.sampler)} samples")
print(f"notMNIST Test set: {len(notMNIST_dataset.test_loader.dataset)} samples")
print()

# now determine the smallest test set size
test_set_sizes = [len(MNIST_dataset.test_loader.dataset), len(FMNIST_dataset.test_loader.dataset), len(EMNIST_dataset.test_loader.dataset), len(KMNIST_dataset.test_loader.dataset), len(notMNIST_dataset.test_loader.dataset)]
min_full_test_set_size = min(test_set_sizes)
print(f"The smallest test set size is: {min_full_test_set_size}")



w_learning_rate = 1e-2
h_learning_rate = 1e-2
T = 10

# create a dummy model to see if the model is working
model = Model(
    input_dim=784,
    hidden_dim=512,
    output_dim=10,
    nm_layers=3,
    act_fn=jax.nn.gelu
)

# print the model
print(model)
# now print number of parameters in the model
print(f"Number of parameters: {model.num_parameters()}")


nm_epochs = 75

# Initialize the model and optimizers
with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(jax.numpy.zeros((batch_size, model._dim_input)), None, model=model)
    optim_h = pxu.Optim(optax.sgd(h_learning_rate), pxu.Mask(pxc.VodeParam)(model))
    optim_w = pxu.Optim(optax.sgd(w_learning_rate, momentum=0.95), pxu.Mask(pxnn.LayerParam)(model))

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for e in range(nm_epochs):

    # train the model
    train(MNIST_dataset.train_loader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
    
    # evaluate the model and get accuracies and losses
    a_train, ys_train, e_train = eval(MNIST_dataset.train_loader, model=model)
    a_val, ys_val, e_val = eval(MNIST_dataset.val_loader, model=model)

    train_losses.append(e_train)
    val_losses.append(e_val)
    train_accuracies.append(a_train)
    val_accuracies.append(a_val)

    print(f"Epoch {e+1}/{nm_epochs} - Train Acc: {a_train:.4f} - Val Acc: {a_val:.4f} - Train Loss: {e_train:.4f} - Val Loss: {e_val:.4f}")

# ID test accuracy
a_test, ys_test, e_test = eval(MNIST_dataset.test_loader, model=model)
print(f"Test Acc: {a_test:.4f} - Test Loss: {e_test:.4f}")

# OOD test accuracy
OOD_a_test, OOD_ys_test, OOD_e_test = eval(FMNIST_dataset.test_loader, model=model)
print(f"OOD Test Acc: {OOD_a_test:.4f} - OOD Test Loss: {OOD_e_test:.4f}")



class DatasetProcessor:
    def __init__(self, model, dataset, dataset_name, new_batch_size, T_inf=100, optim_h=None):
        self.model = model
        self.dataset = dataset
        self.optim_h = optim_h

        self.data = DatasetProcessorData(dataset_name, T_inf, new_batch_size)

        # Automatically process the dataset upon instantiation
        self.process()

    def create_full_test_loader(self):
        return create_new_test_loader_with_batch_size(
            test_loader=self.dataset.test_loader, 
            new_batch_size=self.data.new_batch_size
        )

    def predict_with_logits(self):
        ys_test, logits_test = predict_with_logits(self.full_test_loader, model=self.model)
        self.data.ys_test_jax = jnp.array(ys_test)
        self.data.logits_test_jax = jnp.array(logits_test)
        self.data.max_softmax_values = self.max_softmax_per_sample(self.data.logits_test_jax)

    def max_softmax_per_sample(self, logits):
        softmax_values = jax.nn.softmax(logits, axis=-1)
        max_softmax_values = jnp.max(softmax_values, axis=-1)
        return max_softmax_values

    def run_inference_steps(self):
        self.data.e_pre, self.data.e_post = eval_inference_on_batch(
            self.data.T_inf, 
            self.data.X_test_jax, 
            jax.nn.one_hot(self.data.ys_test_jax, 10), 
            model=self.model, 
            optim_h=self.optim_h
        )

    def compute_probs(self):
        self.data.p_pre = jnp.exp(-self.data.e_pre)
        self.data.p_post = jnp.exp(-self.data.e_post)

    def total_likelihood_from_energies(self):
        F_pre = jnp.sum(self.data.e_pre)
        F_post = jnp.sum(self.data.e_post)
        self.data.NLL_pre = F_pre
        self.data.NLL_post = F_post
        self.data.likelihood_pre = jnp.exp(-F_pre)
        self.data.likelihood_post = jnp.exp(-F_post)

    def process(self):
        self.model.clear_params(pxc.VodeParam)
        self.full_test_loader = self.create_full_test_loader()
        self.data.X_test_jax = next(iter(self.full_test_loader))[0]
        self.predict_with_logits()
        self.run_inference_steps()
        self.compute_probs()
        self.total_likelihood_from_energies()

    def get_results(self):
        return {
            'X_test': self.data.X_test_jax,
            'ys_test': self.data.ys_test_jax,
            'logits_test': self.data.logits_test_jax,
            'max_softmax_values': self.data.max_softmax_values,
            'e_pre': self.data.e_pre,
            'e_post': self.data.e_post,
            'p_pre': self.data.p_pre,
            'p_post': self.data.p_post,
            'NLL_pre': self.data.NLL_pre,
            'NLL_post': self.data.NLL_post,
            'likelihood_pre': self.data.likelihood_pre,
            'likelihood_post': self.data.likelihood_post,
        }

    def save_data(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.data, f)

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        

# Instantiate the DatasetProcessor for all datasets mnist, kmnist, fmnist, emnist, notmnist
mnist_DP = DatasetProcessor(
    model=model, 
    dataset=MNIST_dataset, 
    dataset_name="mnist", 
    new_batch_size=min_full_test_set_size, 
    optim_h=optim_h
)

kmnist_DP = DatasetProcessor(
    model=model, 
    dataset=KMNIST_dataset, 
    dataset_name="kmnist", 
    new_batch_size=min_full_test_set_size, 
    optim_h=optim_h
)

fmnist_DP = DatasetProcessor(
    model=model, 
    dataset=FMNIST_dataset, 
    dataset_name="fmnist", 
    new_batch_size=min_full_test_set_size, 
    optim_h=optim_h
)

emnist_DP = DatasetProcessor(
    model=model, 
    dataset=EMNIST_dataset, 
    dataset_name="emnist", 
    new_batch_size=min_full_test_set_size, 
    optim_h=optim_h
)

notmnist_DP = DatasetProcessor(
    model=model, 
    dataset=notMNIST_dataset,
    dataset_name="notmnist", 
    new_batch_size=min_full_test_set_size, 
    optim_h=optim_h
)


# Define the directory where you want to save the files
save_dir = './data/energy'

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save each processor's data
mnist_DP.save_data(os.path.join(save_dir, 'mnist_DP.pkl'))
kmnist_DP.save_data(os.path.join(save_dir, 'kmnist_DP.pkl'))
fmnist_DP.save_data(os.path.join(save_dir, 'fmnist_DP.pkl'))
emnist_DP.save_data(os.path.join(save_dir, 'emnist_DP.pkl'))
notmnist_DP.save_data(os.path.join(save_dir, 'notmnist_DP.pkl'))
