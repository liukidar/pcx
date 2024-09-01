import os
import sys
import itertools
import json
import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.functional as pxf
import pcax.utils as pxu

# 3rd party imports
import jax
from jax import jit
import jax.numpy as jnp
import optax

# Custom imports
import causal_helpers
from causal_helpers import simulate_dag, simulate_parameter, simulate_linear_sem
from causal_helpers import load_adjacency_matrix, set_random_seed, plot_adjacency_matrices

# Set random seed
seed = 23
set_random_seed(seed)

# Causal libraries
import cdt, castle, dodiscover
from dodiscover.toporder import DAS, SCORE, CAM
from dodiscover import make_context
from cdt.metrics import precision_recall, SHD, SID
from castle.metrics import MetricsDAG
from castle.common import GraphDAG

# Load adjacency matrices
folder = "./data/"
G_A_init_t_ordered_adj_matrix = load_adjacency_matrix(os.path.join(folder, 'G_A_init_t_ordered_adj_matrix.npy'))
G_A_init_t_ordered_dag_adj_matrix = load_adjacency_matrix(os.path.join(folder, 'G_A_init_t_ordered_dag_adj_matrix.npy'))
ER = load_adjacency_matrix(os.path.join(folder, 'ER_adj_matrix.npy'))
ER_dag = load_adjacency_matrix(os.path.join(folder, 'ER_dag_adj_matrix.npy'))

# Change names for connectome matrices
C = G_A_init_t_ordered_adj_matrix
C_dag = G_A_init_t_ordered_dag_adj_matrix

# Ensure matrices are binary
ER_dag_bin = (ER_dag != 0).astype(int)
C_dag_bin = (C_dag != 0).astype(int)
SF_true = simulate_dag(d=279, s0=1674, graph_type='SF')

# Select B_true graph (the true adjacency matrix to use)
#B_true = C_dag_bin  # if you want to use the connectome-based DAG
#B_true = ER_dag_bin # if you want to use the ER-based DAG
B_true = SF_true # if you want to use the SF-based DAG

# Simulate parameters and data
W_true = simulate_parameter(B_true)
X = simulate_linear_sem(W_true, n=1000, sem_type='gauss')

################################################# HYPERPARAMETERS #################################################

# Set hyperparameter ranges for grid search
w_learning_rate_values = [5e-4, 5e-3, 5e-2]
lam_h_values = [1e1, 1e2, 1e3]
lam_l1_values = [1e-4, 1e-3, 1e-2]
h_learning_rate = 5e-4
T = 1
nm_epochs = 100000
batch_size = 128

# Determine the graph type based on B_true
if np.array_equal(B_true, C_dag_bin):
    graph_type = 'connectome'
elif np.array_equal(B_true, ER_dag_bin):
    graph_type = 'ER'
elif np.array_equal(B_true, SF_true):
    graph_type = 'SF'
else:
    graph_type = 'unknown'

num_samples = X.shape[0]
n_nodes = X.shape[1]

################################################# MODEL DEFINITION #################################################

class Complete_Graph(pxc.EnergyModule):
    def __init__(self, input_dim: int, n_nodes: int, has_bias: bool = False) -> None:
        super().__init__()

        self.input_dim = px.static(input_dim)  # Ensure input_dim is static
        self.n_nodes = px.static(n_nodes)  # Keep n_nodes as a static value
        self.has_bias = has_bias

        # Initialize a single linear layer for the weights and wrap it in a list
        self.layers = [pxnn.Linear(n_nodes * input_dim, n_nodes * input_dim, bias=has_bias)]
        
        # Zero out the diagonal weights to avoid self-loops
        weight_matrix = self.layers[0].nn.weight.get()
        weight_matrix = weight_matrix.reshape(n_nodes, input_dim, n_nodes, input_dim)
        for i in range(n_nodes):
            weight_matrix = weight_matrix.at[i, :, i, :].set(jnp.zeros((input_dim, input_dim)))
        self.layers[0].nn.weight.set(weight_matrix.reshape(n_nodes * input_dim, n_nodes * input_dim))

        # Initialize vodes as a list containing a single matrix
        self.vodes = [pxc.Vode((n_nodes, input_dim))]

    def freeze_nodes(self, freeze=True):
        self.vodes[0].h.frozen = freeze

    def are_vodes_frozen(self):
        """Check if all vodes in the model are frozen."""
        return self.vodes[0].h.frozen
    
    def get_W(self):
        """This function returns the weighted adjacency matrix based on the linear layer in the model."""
        W = self.layers[0].nn.weight.get()
        W_T = W.T
        return W_T
    
    def save(self, file_name):
        pxu.save_params(self, file_name)

    def __call__(self, x=None):
        n_nodes = self.n_nodes.get()
        input_dim = self.input_dim.get()
        if x is not None:
            # print the shape of x before reshaping when x is not None
            #print("The shape of x before reshaping when x is not None: ", x.shape)

            # Initialize nodes with given data
            reshaped_x = x.reshape(n_nodes, input_dim)

            # print the shape of reshaped_x when x is not None
            #print("The shape of reshaped_x when x is not None: ", reshaped_x.shape)

            self.vodes[0](reshaped_x)
        else:
            # Perform forward pass using stored values
            #x_ = self.vodes[0].get('h').reshape(n_nodes * input_dim, 1)

            x_ = self.vodes[0].get('h')
            # print the shape of x_ when x is None before reshaping
            #print("The shape of x_ when x is None before reshaping: ", x_.shape)

            x_ = x_.reshape(n_nodes * input_dim, 1)
            # print the shape of x_ when x is None after reshaping
            #print("The shape of x_ when x is None after reshaping: ", x_.shape)

            # Perform the matrix-matrix multiplication
            #output = self.layers[0](x_).reshape(n_nodes, input_dim)

            output = self.layers[0](x_)
            # print the shape of output before reshaping
            #print("The shape of output before reshaping: ", output.shape)
            #output = output.reshape(n_nodes, input_dim)
            # print the shape of output after reshaping
            #print("The shape of output after reshaping: ", output.shape)

            # Set the new values in vodes
            self.vodes[0](output)

        # Return the output directly
        return self.vodes[0].get('h')

# Training and evaluation functions
@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), in_axes=(0,), out_axes=0)
def forward(x, *, model: Complete_Graph):
    return model(x)

@pxf.vmap(pxu.Mask(pxc.VodeParam | pxc.VodeParam.Cache, (None, 0)), out_axes=(None, 0), axis_name="batch")
def energy(*, model: Complete_Graph):
    x_ = model(None)
    
    W = model.get_W()
    d = model.n_nodes.get()

    # compute the PC energy term (loss) - equivalent to negative log-likelihood
    # thus no need to consider the logdet term: -jnp.linalg.slogdet(jnp.eye(d) - W)[1]
    energy = model.energy()

    # compute L1 regularization term of W (not normalized by batch size)
    l1_reg = jnp.sum(jnp.abs(W))

    # compute the DAG penalty term using the matrix exponential (h_reg)
    h_reg = jnp.trace(jax.scipy.linalg.expm(W * W)) - d
    
    # Combine loss, soft DAG constraint, and L1 regularization
    obj = jax.lax.pmean(energy, axis_name="batch") + lam_h * h_reg + lam_l1 * l1_reg

    return obj, x_

@pxf.jit(static_argnums=0)
def train_on_batch(T: int, x: jax.Array, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):

    print("Training!")  # this will come in handy later

    # This only sets an internal flag to be "train" (instead of "eval")
    model.train()

    # freeze nodes at the start of training
    model.freeze_nodes(freeze=True)

    # init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, model=model)
        
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.Mask(pxnn.LayerParam, [False, True]), has_aux=True)(energy)(model=model)
        
        # Zero out the diagonal gradients using jax.numpy.fill_diagonal
        weight_grads = g["model"].layers[0].nn.weight.get()
        weight_grads = jax.numpy.fill_diagonal(weight_grads, 0.0, inplace=False)
        # print the grad values using the syntax jax.debug.print("🤯 {x} 🤯", x=x)
        #jax.debug.print("{weight_grads}", weight_grads=weight_grads)
        g["model"].layers[0].nn.weight.set(weight_grads)

    optim_w.step(model, g["model"])

    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        forward(None, model=model)
        e_avg_per_sample = model.energy() # this returns average value of objective per sample in the batch

    # unfreeze nodes at the end of training
    model.freeze_nodes(freeze=False)

    return e_avg_per_sample

def train(dl, T, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):
    e_avg_per_sample_energies = []
    for batch in dl:

        e_avg_per_sample = train_on_batch(T, batch, model=model, optim_w=optim_w, optim_h=optim_h)
        e_avg_per_sample_energies.append(e_avg_per_sample)

    W = model.get_W()

    # compute epoch energy
    epoch_energy = jnp.mean(jnp.array(e_avg_per_sample_energies))
    return W, epoch_energy

@jit
def MAE(W_true, W):
    """This function returns the Mean Absolute Error for the difference between the true weighted adjacency matrix W_true and th estimated one, W."""
    MAE_ = jnp.mean(jnp.abs(W - W_true))
    return MAE_

def compute_binary_adjacency(W, threshold=0.3):
    """
    Compute the binary adjacency matrix by thresholding the input matrix.

    Args:
    - W (array-like): The weighted adjacency matrix (can be a JAX array or a NumPy array).
    - threshold (float): The threshold value to determine the binary matrix. Default is 0.3.

    Returns:
    - B_est (np.ndarray): The binary adjacency matrix where each element is True if the corresponding 
                          element in W is greater than the threshold, otherwise False.
    """
    # Convert JAX array to NumPy array if necessary
    if isinstance(W, jnp.ndarray):
        W = np.array(W)

    # Compute the binary adjacency matrix
    B_est = np.array(np.abs(W) > threshold)
    
    return B_est

# for reference compute the MAE, SID, and SHD between the true adjacency matrix and an all-zero matrix and then print it
# this acts as a baseline for the MAE, SID, and SHD similar to how 1/K accuracy acts as a baseline for classification tasks where K is the number of classes
W_zero = np.zeros_like(W_true)
print("MAE between the true adjacency matrix and an all-zero matrix: ", MAE(W_true, W_zero))
print("SHD between the true adjacency matrix and an all-zero matrix: ", SHD(B_true, compute_binary_adjacency(W_zero)))
#print("SID between the true adjacency matrix and an all-zero matrix: ", SID(W_true, W_zero))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

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

################################################# GRID SEARCH LOOP #################################################

# Grid search loop over all combinations of hyperparameters
for w_learning_rate, lam_h, lam_l1 in itertools.product(w_learning_rate_values, lam_h_values, lam_l1_values):

    # Create a unique experiment folder based on the hyperparameters
    experiment_name = f'exp_lrW_{w_learning_rate}_lrH_{h_learning_rate}_T_{T}_epochs_{nm_epochs}_bs_{batch_size}_lamH_{lam_h}_lamL1_{lam_l1}_{graph_type}_{num_samples}samples'
    experiment_folder = os.path.join('experiments', experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Set paths for saving outputs
    log_file_path = os.path.join(experiment_folder, 'run_log.txt')
    hyperparams_path = os.path.join(experiment_folder, 'hyperparams.json')
    metrics_json_path = os.path.join(experiment_folder, 'metrics.json')
    arrays_path = os.path.join(experiment_folder, 'array_values.npz')
    figures_path = os.path.join(experiment_folder, 'figures')
    model_path = os.path.join(experiment_folder, 'model')
    os.makedirs(figures_path, exist_ok=True)

    # Redirect output to a log file
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file

    # Define model and training setup
    model = Complete_Graph(input_dim=1, n_nodes=n_nodes, has_bias=False)

    # Define the data loader
    dataset = CustomDataset(X)
    dl = TorchDataloader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizers
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, model.n_nodes.get())), model=model)
        optim_h = pxu.Optim(optax.sgd(h_learning_rate), pxu.Mask(pxc.VodeParam)(model))
        optim_w = pxu.Optim(optax.adam(w_learning_rate), pxu.Mask(pxnn.LayerParam)(model))

    # Initialize lists to store metrics
    MAEs = []
    SHDs = []
    energies = []

    # Calculate the initial MAE, SID, and SHD
    MAE_init = MAE(W_true, model.get_W())
    print(f"Start difference (cont.) between W_true and W_init: {MAE_init:.4f}")

    SHD_init = SHD(B_true, compute_binary_adjacency(model.get_W()))
    print(f"Start SHD between W_true and W_init: {SHD_init:.4f}")

    # Print the values of the diagonal of the initial W
    print("The diagonal of the initial W: ", jnp.diag(model.get_W()))

    # Start timing
    start_time = timeit.default_timer()

    # Training loop with early stopping based on average SHD
    with tqdm(range(nm_epochs), position=0, leave=True) as pbar:
        for epoch in pbar:
            # Train for one epoch
            W, epoch_energy = train(dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)

            # Calculate metrics
            W = np.array(W)
            current_mae = float(MAE(W_true, W))
            current_shd = float(SHD(B_true, compute_binary_adjacency(W)))
            current_energy = float(epoch_energy)

            # Store metrics
            MAEs.append(current_mae)
            SHDs.append(current_shd)
            energies.append(current_energy)

            # Check for average SHD improvement
            if epoch >= 6000:
                recent_avg_shd = np.mean(SHDs[-3000:])
                previous_avg_shd = np.mean(SHDs[-6000:-3000])

                if recent_avg_shd >= previous_avg_shd:
                    print("Stopping early due to no improvement in average SHD over the last 3000 epochs.")
                    break

            # Update progress bar
            pbar.set_description(f"MAE {current_mae:.4f}, SHD {current_shd:.4f} || Energy {current_energy:.4f}")

    # End timing
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    average_time_per_epoch = total_time / max(len(SHDs), 1)

    # Print the running times
    print(f"An epoch (with compiling and testing) took on average: {average_time_per_epoch:.4f} seconds")
    print(f"Total running time: {total_time:.4f} seconds")

    # Save results and model
    hyperparameters = {
        'w_learning_rate': w_learning_rate,
        'h_learning_rate': h_learning_rate,
        'T': T,
        'nm_epochs': nm_epochs,
        'batch_size': batch_size,
        'lam_h': lam_h,
        'lam_l1': lam_l1,
        'graph_type': graph_type,
        'edges_per_node': np.sum(B_true) / n_nodes,
        'num_samples': num_samples,
        'total_time': total_time,
        'average_time_per_epoch': average_time_per_epoch,
        'd': n_nodes,
    }
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Save metrics dictionary as JSON
    with open(metrics_json_path, 'w') as json_file:
        metrics = {'MAEs': MAEs, 'SHDs': SHDs, 'energies': energies}
        json.dump(metrics, json_file, indent=4)

    # Save other arrays and values in .npz file
    np.savez(arrays_path, MAEs=MAEs, SHDs=SHDs, energies=energies, B_true=B_true)

    # Plot metrics and save figures
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Metrics Over Epochs', fontsize=16, weight='bold')

    # Plot MAE
    sns.lineplot(x=range(len(MAEs)), y=MAEs, ax=axs[0], color=palette[0])
    axs[0].set_title("Mean Absolute Error (MAE)", fontsize=14)
    axs[0].set_xlabel("Epoch", fontsize=12)
    axs[0].set_ylabel("MAE", fontsize=12)
    axs[0].grid(True)

    # Plot SHD
    sns.lineplot(x=range(len(SHDs)), y=SHDs, ax=axs[1], color=palette[2])
    axs[1].set_title("Structural Hamming Distance (SHD)", fontsize=14)
    axs[1].set_xlabel("Epoch", fontsize=12)
    axs[1].set_ylabel("SHD", fontsize=12)
    axs[1].grid(True)

    # Plot Energy
    sns.lineplot(x=range(len(energies)), y=energies, ax=axs[2], color=palette[3])
    axs[2].set_title("Energy", fontsize=14)
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel("Energy", fontsize=12)
    axs[2].grid(True)

    # Save combined metrics plot
    fig.savefig(os.path.join(figures_path, 'metrics_over_epochs.png'))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Save adjacency matrices plot
    W_est = np.array(model.get_W())
    B_est = compute_binary_adjacency(W_est, threshold=0.3)
    combined_adj_matrix_path = os.path.join(figures_path, 'combined_graph_comparison.png')
    plot_adjacency_matrices(B_true, B_est, save_path=combined_adj_matrix_path)

    # Plot and save GraphDAG
    GraphDAG(B_est, B_true)
    gcastle_graph_path = os.path.join(figures_path, 'graph_dag_comparison_gcastle.png')
    plt.gcf().savefig(gcastle_graph_path, bbox_inches='tight')
    plt.show()

    # Calculate and save metrics using MetricsDAG
    met_pcax = MetricsDAG(B_est, B_true)
    print(met_pcax.metrics)

    # Save the model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(os.path.join(model_path, 'model'))
    print(f'Model saved to {model_path}/model')

    # Close log file
    log_file.close()
    sys.stdout = sys.__stdout__  # Reset standard output


# Example code to load the model later
# model = Complete_Graph(input_dim, n_nodes, has_bias=False)  # Reinitialize the model structure
# model.load(os.path.join(model_path, 'model'))
# print('Model loaded successfully')
