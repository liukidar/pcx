# built-in
import json
import time
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 3rd party
import numpy as np
import matplotlib.pyplot as plt

# own
from helpers import get_dataloaders, Progress

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


def get_model_by_name(model_name):
    models = [TwoLayerNN]
    for model in models:
        if model.name() == model_name:
            return model
    raise ValueError(f"Model {model_name} not found")


class TwoLayerNN(px.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, subset_size: int, act_fn: Callable[[jax.Array], jax.Array]) -> None:
        super().__init__()
        self._train_step = 0
        self._val_step = 0
        self._test_step = 0

        # required for weight initialization scenarios
        self._dim_output = output_dim
        self._subset_size = subset_size

        self.act_fn = px.static(act_fn)

        self.layers = [
            pxnn.Linear(input_dim, hidden_dim),
            pxnn.Linear(hidden_dim, output_dim)
        ]

    @staticmethod
    def name():
        return "two_layer_nn"

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

    def epoch_step(self):
        self._train_step = 0
        self._val_step = 0

    def initialize_with_previous_weights(self, previous_weights):
            # target_weights: the hidden weights of the current model: self.layers[0].nn.weight
            # previous_weights: the hidden weights of the previous model: prev_model.layers[0].nn.weight
            self.layers[0].nn.weight.set(self.layers[0].nn.weight.at[:previous_weights.shape[0], :previous_weights.shape[1]].set(previous_weights))

    def initialize_with_previous_bias(self, previous_bias):
            # target_bias: the hidden bias of the current model: self.layers[0].nn.bias
            # previous_bias: the hidden bias of the previous model: prev_model.layers[0].nn.bias
            self.layers[0].nn.bias.set(self.layers[0].nn.bias.at[:previous_bias.shape[0]].set(previous_bias))

    def init_weights(self, prev_model):
        if prev_model is None or self.num_parameters() > self._subset_size * self._dim_output:
            # Initialize smallest net using Xavier Glorot-uniform distribution

            # create a glorot uniform initializer:
            # see: https://pytorch.org/docs/2.0/nn.init.html?highlight=xavier#torch.nn.init.xavier_uniform_
            # see: https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
            #initializer = jax.nn.initializers.glorot_uniform() # this is wrong, due to the scale being 1.0 by default for this function
            # ReLU adjusting scale value in in JAX
            scale_ = 6.0
            initializer_ = jax.nn.initializers.variance_scaling(scale=scale_, mode='fan_avg', distribution='uniform')
            # more here: https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
            # Apply Glorot uniform initialization to the weights only
            for l in self.layers:
                l.nn.weight.set(initializer_(px.RKG(), l.nn.weight.shape))

            # debug print statement
            # print that the model was initialized with Glorot uniform
            """
            print(f"Initialized model with Glorot uniform initialization")
            """            

        else:
            # Initialize larger net using previous model weights and biases
            # Remaining weights are initialized using normal distribution with mean 0 and std 0.01
            initializer_ = jax.nn.initializers.normal(stddev=0.01)

            # Initialize all weight matrices with normal distribution with mean 0 and std 0.01
            for l in self.layers:
                l.nn.weight.set(initializer_(px.RKG(), l.nn.weight.shape))

            # Partially initialize the weights and biases of the hidden layer with the previous model's weights and biases
            self.initialize_with_previous_weights(prev_model.layers[0].nn.weight)
            self.initialize_with_previous_bias(prev_model.layers[0].nn.bias)

            # debug print statements
            """
            # print that the model was initialized with previous model's hidden weights and hidden biases
            print(f"Initialized model with previous model's hidden weights and hidden biases")
            # compare previous weights with current weights and previous biases with current biases and return boolean for both cases
            hidden_W = self.layers[0].nn.weight.get()[:prev_model.layers[0].nn.weight.shape[0], :prev_model.layers[0].nn.weight.shape[1]]
            previous_W = prev_model.layers[0].nn.weight.get()[:prev_model.layers[0].nn.weight.shape[0], :prev_model.layers[0].nn.weight.shape[1]]
            # Verify the replacement
            print("Are the weights the same?", jnp.all(hidden_W == previous_W))

            hidden_b = self.layers[0].nn.bias.get()[:prev_model.layers[0].nn.bias.shape[0]]
            previous_b = prev_model.layers[0].nn.bias.get()[:prev_model.layers[0].nn.bias.shape[0]]
            # Verify the replacement
            print("Are the biases the same?", jnp.all(hidden_b == previous_b))
            """

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        x = self.layers[-1](x)

        return x

def ce_loss(output, one_hot_label):
    return -one_hot_label * jax.nn.log_softmax(output)


@pxf.vmap({"model": None}, in_axes=0, out_axes=0)
def forward(x, *, model: TwoLayerNN):
    return model(x)


@pxf.vmap({"model": None}, in_axes=(0, 0), out_axes=(None, 0), axis_name="batch")
def loss(x, y, *, model: TwoLayerNN):
    y_ = model(x)
    return jax.lax.pmean(ce_loss(y_, y).sum(), "batch"), y_


@pxf.jit()
def train_on_batch(x: jax.Array, y: jax.Array, *, model: TwoLayerNN, optim_w: pxu.Optim):
    model.train()

    with pxu.step(model):
        (e, y_), g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(loss)(x, y, model=model)
    optim_w.step(model, g["model"])


def train(dl, *, model: TwoLayerNN, optim_w: pxu.Optim, progress: Progress):
    for x, y in dl:
        progress.update_batch()
        train_on_batch(x, jax.nn.one_hot(y, 10), model=model, optim_w=optim_w)


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: TwoLayerNN):
    model.eval()

    with pxu.step(model):
        e, y_ = loss(x, y, model=model)
        y_ = y_.argmax(axis=-1)

    return (y_ == y).mean(), y_, e


def eval(dl, *, model: TwoLayerNN):
    acc = []
    es = []
    ys_ = []
    for x, y in dl:
        a, y_, e = eval_on_batch(x, jax.nn.one_hot(y, 10), model=model)
        acc.append(a)
        es.append(e)
        ys_.append(y_)

    return float(np.mean(acc)), np.concatenate(ys_), float(np.mean(es))

###################################### end of model related code ##########################################

class Training:
    def __init__(self, model, dataset, num_params, epochs, batch_size, learning_rate, noise_level, prev_model=None, num_models=None):
        self._train_loader = dataset.train_loader
        self._subset_size = len(dataset.train_loader.sampler)
        self._val_loader = dataset.val_loader
        self._test_loader = dataset.test_loader
        self.num_classes = len(np.unique(dataset.train_loader.dataset.targets))
        self.input_dim = self._train_loader.dataset[0][0].shape[0]        
        self._interpolation_threshold = self.num_classes * len(dataset.train_loader.sampler)
        self._all_models = []
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._noise_level = noise_level
        self.all_train_losses = {}
        self.all_val_losses = {}
        self.all_test_losses = {}
        self._dataset_name = type(self._val_loader.dataset).__name__
        self._model_name = model.name()
        self._loss_name = "CrossEntropyLoss"
        self._prev_model = None

        if num_models is not None:
            num_params = num_params[:num_models]

        # in this loop we create all models with different hidden layer sizes
        for p in num_params:
            self._all_models.append(TwoLayerNN(input_dim=self.input_dim, hidden_dim=p, output_dim=self.num_classes, 
                                               subset_size=self._subset_size, act_fn=jax.nn.relu))


    def save(self):
        path = os.path.join("data", "results_bp", self._model_name, self._loss_name)
        file_name = os.path.join(path, f"epochs_{self._epochs}_bs_{self._batch_size}_lr_{self._learning_rate}_noise_{self._noise_level}.json")
        if not os.path.exists(path):
            os.makedirs(path)
        content = {}
        content["Train losses"] = self.all_train_losses
        content["Val losses"] = self.all_val_losses
        content["Test losses"] = self.all_test_losses
        if os.path.exists(file_name):
            with open(file_name, "r") as fd:
                old_content = json.load(fd)
            for loss_key in old_content.keys():
                for model_key in old_content[loss_key].keys():
                    content[loss_key][model_key] = old_content[loss_key][model_key]
        with open(file_name, "w") as fd:
            json.dump(content, fd, indent=4, sort_keys=True)

    def start(self):
        try:
            _, _ = os.popen("stty size", "r").read().split()
            run_from_term = True
        except Exception:
            run_from_term = False
        progress = Progress(len(self._all_models), self._epochs, len(self._train_loader), run_from_term)
        progress.init_print(len(self._all_models), self._model_name, self._dataset_name)


        for model in self._all_models:  # different sized models

            # Initialize the model with the previous model's weights if it exists
            model.init_weights(self._prev_model)

            # Initialize the optimizer
            with pxu.step(model):
                optim_w = pxu.Optim(optax.sgd(self._learning_rate, momentum=0.95), pxu.Mask(pxnn.LayerParam)(model))

            progress.update_model()

            train_losses = []
            val_losses = []
            for e in range(self._epochs):
                model.epoch_step()
                # train the model
                train(dataset.train_loader, model=model, optim_w=optim_w, progress=progress)
                
                # evaluate the model and get accuracies and losses
                a_train, ys_train, e_train = eval(dataset.train_loader, model=model)
                a_val, ys_val, e_val = eval(dataset.val_loader, model=model)

                train_losses.append(e_train)
                val_losses.append(e_val)

                progress.update_epoch(train_losses[-1], val_losses[-1])

                # TODO: only for two layer nn?
                if train_losses[-1] == 0.0 and model.num_parameters() < self._interpolation_threshold:
                    break

            a_test, ys_test, e_test = eval(dataset.test_loader, model=model)

            progress.finished_model(model.num_parameters(), e_test, a_test)

            model_name = str(model.num_parameters())
            self.all_test_losses[model_name] = e_test
            self.all_train_losses[model_name] = train_losses
            self.all_val_losses[model_name] = val_losses
            self._prev_model = model            

            # Explicitly create the path variable with additional parameters
            path = os.path.join(
                "data", 
                "models_bp", 
                self._model_name, 
                self._loss_name, 
                f"epochs_{self._epochs}_bs_{self._batch_size}_lr_{self._learning_rate}_noise_{self._noise_level}"
            )            
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = os.path.join(path, model_name)
            self.save()
            print(f"Saving model to \"{file_name}\"\n")
            model.save(file_name)
        progress.finished_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple trainings with growing network capacity")

    parser.add_argument("--num", type=int, help="the number of model sizes to run, starting at smallest model or model given with --prev flag", dest="num")
    parser.add_argument("--config", type=str, help="config file for training", required=True, dest="config")
    parser.add_argument("--epochs", type=int, help="The number of epochs to train, if not given, the value in the config file is taken", dest="epochs")
    parser.add_argument("--batch_size", type=int, help="The batch size for training", dest="batch_size", default=128)
    parser.add_argument("--learning_rate", type=float, help="The SGD learning rate", dest="learning_rate", default=0.01)
    parser.add_argument("--noise_level", type=float, help="The noise level for label noise", dest="noise_level", default=0.2)

    args = parser.parse_args()

    with open(args.config, "r") as fd:
        config = json.load(fd)

    dataset_name = "mnist"
    model_name = "two_layer_nn"
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    noise_level = args.noise_level
    train_subset_size = config["train_subset_size"][model_name.lower()]
    epochs = args.epochs if args.epochs else config["epochs"]
    num_params = config["num_params"][model_name]
    previous_model = None
    num_models = None
    if args.num:
        num_models = args.num

    model = get_model_by_name(model_name)
    dataset = get_dataloaders(dataset_name, train_subset_size, batch_size, noise_level)

    training = Training(model, dataset, num_params, epochs, batch_size, learning_rate, noise_level, prev_model=previous_model, num_models=num_models)
    training.start()
    training.save()
