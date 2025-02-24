import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run experiments on a single folder.")
parser.add_argument("--n_samples", type=int, default=2000,
                    help="Number of training samples to use (to select the train_n_samples_<n>.csv file).")
parser.add_argument("--num_seeds", type=int, default=10,
                    help="Number of seeds to run experiments for.")
parser.add_argument("--gpu", type=int, default=1, help="GPU index to use")
args = parser.parse_args()

# Set GPU-related environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cuda"  # Force JAX to use CUDA

import pandas as pd
import json
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import timeit
import optuna

import matplotlib.pyplot as plt
import seaborn as sns
import gc # garbage collection

###########################################################

def check_and_log_cycles(B, model_name):
    """Check if B is a DAG and log issues."""
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(G):
        print(f"🚨 Warning: {model_name} produced a cyclic B_est!")
        print(f"Cycle: {list(nx.find_cycle(G))}")
        return False  # Indicates a cycle is present
    return True  # No cycles


# ------------------ Global Parameters ------------------

num_samples = args.n_samples  # used for loading training data
w_learning_rate = 1e-3
h_learning_rate = 1e-4
nm_epochs = 2500
batch_size = 256
every_n_epochs = nm_epochs // 5
seed_values = list(range(1, args.num_seeds + 1))


# List of eligible base folders for mixed data experiments.
# Each base folder contains the global files (adj_matrix.csv, W_adj_matrix.csv)
# and a set of subdirectories named "n_disc_*" with training files.
# Define the eligible mixed data base folders (do NOT include "experiment_results")
mixed_base_folders = [
    "/share/amine.mcharrak/mixed_data_final/15ER60_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/20ER80_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/20SF80_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/15SF60_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/15WS60_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/10ER40_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/10SF40_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/20WS80_linear_mixed_seed_1",
    "/share/amine.mcharrak/mixed_data_final/10WS40_linear_mixed_seed_1"
]

def run_experiment(seed, train_file_path, lam_h_val, lam_l1_val):
    global lam_h, lam_l1
    # Override global hyperparameters used in the loss.
    lam_h = lam_h_val
    lam_l1 = lam_l1_val

    print(f"Running experiment on {train_file_path} with lam_h={lam_h_val}, lam_l1={lam_l1_val}")    

    # The global matrices (adjacency, weight) are in the parent directory of the n_disc_* folder.
    base_folder = os.path.dirname(os.path.dirname(train_file_path))
    adj_matrix = pd.read_csv(os.path.join(base_folder, 'adj_matrix.csv'), header=None)
    weighted_adj_matrix = pd.read_csv(os.path.join(base_folder, 'W_adj_matrix.csv'), header=None)
    
    # Load training data.
    X_df = pd.read_csv(train_file_path, header=None)
    B_true = adj_matrix.values
    W_true = weighted_adj_matrix.values
    X = X_df.values

    X_df.head()
    # show unique values in each column
    print(X_df.nunique())
    # Determine if each variable is continuous or discrete based on the number of unique values
    is_cont_node = np.array([True if X_df[col].nunique() > 2 else False for col in X_df.columns])
    is_cont_node = is_cont_node.tolist()
    print(is_cont_node)

    # print how many non-zero entries are in the true DAG
    print(f"Number of non-zero entries in the true DAG: {np.count_nonzero(B_true)}")

    # Usage
    input_dim = 1
    n_samples = X.shape[0]
    n_nodes = X.shape[1]
    stddev = 1/n_nodes

    #model = Complete_Graph(input_dim, n_nodes, has_bias=True, is_cont_node=is_cont_node, seed=seed, stddev=stddev)
    model = Complete_Graph(input_dim, n_nodes, has_bias=True, is_cont_node=is_cont_node, seed=seed, stddev=stddev)

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


    # Create the custom dataset
    dataset = CustomDataset(X)
    # Create the dataloader
    dl = TorchDataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, prefetch_factor=None, persistent_workers=False)

    # Training an1d evaluation functions
    @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=0)
    def forward(x, *, model: Complete_Graph):
        return model(x)

    # @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, 0), axis_name="batch") # if only one output
    @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), out_axes=(None, None, None, None, 0), axis_name="batch") # if multiple outputs
    def energy(*, model: Complete_Graph):

        x_ = model(None)
        
        W = model.get_W()
        # Dimensions of W
        d = W.shape[0]

        # PC energy term
        pc_energy = jax.lax.pmean(model.energy(), axis_name="batch")

        # L1 regularization using adjacency matrix (scaled by Frobenius norm)
        l1_reg = jnp.sum(jnp.abs(W)) / (jnp.linalg.norm(W, ord='fro') + 1e-8)
        #l1_reg = jnp.sum(jnp.abs(W)) / d

        # DAG constraint (stable logarithmic form)
        #h_reg = notears_dag_constraint(W)
        #h_reg = notears_dag_constraint(W)/ (jnp.sqrt(d) + 1e-8)
        #h_reg = notears_dag_constraint(W) / d  # with normalization

        #h_reg = dagma_dag_constraint(W)
        h_reg = dagma_dag_constraint(W) / (jnp.sqrt(d) + 1e-8)
        #h_reg = dagma_dag_constraint(W) / d  # with normalization
            
        # Combined loss
        obj = pc_energy + lam_h * h_reg + lam_l1 * l1_reg

        # Ensure obj is a scalar, not a (1,) array because JAX's grad and value_and_grad functions are designed
        # to compute gradients of *scalar-output functions*
        obj = obj.squeeze()  # explicitly converte the (1,) array (obj) to a scalar of shape ()  
        
        return obj, pc_energy, h_reg, l1_reg, x_

    @pxf.jit(static_argnums=0)
    def train_on_batch(T: int, x: jax.Array, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):

        model.train()

        model.freeze_nodes(freeze=True)

        # init step
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):

            forward(x, model=model)

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):

            (obj, (pc_energy, h_reg, l1_reg, x_)), g = pxf.value_and_grad(
                pxu.M(pxnn.LayerParam).to([False, True]), 
                has_aux=True
            )(energy)(model=model) # pxf.value_and_grad returns a tuple structured as ((value, aux), grad), not as six separate outputs.
            
            #print("Gradient structure:", g)

            # Zero out the diagonal gradients using jax.numpy.fill_diagonal
            weight_grads = g["model"].layers[0].nn.weight.get()
            weight_grads = jax.numpy.fill_diagonal(weight_grads, 0.0, inplace=False)
            # print the grad values using the syntax jax.debug.print("🤯 {x} 🤯", x=x)
            #jax.debug.print("{weight_grads}", weight_grads=weight_grads)
            g["model"].layers[0].nn.weight.set(weight_grads)

        optim_w.step(model, g["model"])
        #optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):

            forward(None, model=model)
            e_avg_per_sample = model.energy()


        model.freeze_nodes(freeze=False)


        return pc_energy, l1_reg, h_reg, obj

    def train(dl, T, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):
        batch_pc_energies = []
        batch_l1_regs = []
        batch_h_regs = []
        batch_objs = []
        
        for batch in dl:
            pc_energy, l1_reg, h_reg, obj = train_on_batch(
                T, batch, model=model, optim_w=optim_w, optim_h=optim_h
            )
            batch_pc_energies.append(pc_energy)
            batch_l1_regs.append(l1_reg)
            batch_h_regs.append(h_reg)
            batch_objs.append(obj)

        W = model.get_W()

        # Compute epoch averages
        epoch_pc_energy = jnp.mean(jnp.array(batch_pc_energies))
        epoch_l1_reg = jnp.mean(jnp.array(batch_l1_regs))
        epoch_h_reg = jnp.mean(jnp.array(batch_h_regs))
        epoch_obj = jnp.mean(jnp.array(batch_objs))
        
        return W, epoch_pc_energy, epoch_l1_reg, epoch_h_reg, epoch_obj


    # Initialize the model and optimizers
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, model.n_nodes.get())), model=model)
        optim_h = pxu.Optim(lambda: optax.sgd(h_learning_rate))
        optim_w = pxu.Optim(lambda: optax.adamw(w_learning_rate, nesterov=True), pxu.M(pxnn.LayerParam)(model))
    
    # Training loop
    shd_metrics = []
    start_time = timeit.default_timer()
    for epoch in tqdm(range(nm_epochs)):
        W, ep_pc, ep_l1, ep_h, ep_obj = train(dl, T=1, model=model, optim_w=optim_w, optim_h=optim_h)
        if (epoch + 1) % every_n_epochs == 0 or epoch == 0:
            B_est = compute_binary_adjacency(np.array(W))
            # For SHD, assume compute_SHD_robust returns a tuple (normalized, absolute)
            shd_norm, _ = compute_SHD_robust((B_true > 0).astype(np.int8), B_est)
            shd_metrics.append(shd_norm)
            tqdm.write(f"Epoch {epoch+1}: SHD_normalized = {shd_norm:.4f}")
    end_time = timeit.default_timer()
    print(f"Average epoch time: {(end_time - start_time) / nm_epochs:.4f} seconds")
    
    # After training, compute final shd_normalized.
    W_final = model.get_W()
    B_final = compute_binary_adjacency(np.array(W_final))
    final_shd_norm, _ = compute_SHD_robust((B_true > 0).astype(np.int8),
                                     B_final)
    print(f"Final SHD_normalized: {final_shd_norm}")
    return final_shd_norm

# ------------------ Define the Optuna Objective Function ------------------
def objective(trial):
    import random

    # Sample hyperparameters.
    lam_h_sample = trial.suggest_float("lam_h", 1e-5, 1e5, log=True)
    lam_l1_sample = trial.suggest_float("lam_l1", 1e-5, 1e5, log=True)
    
    metric_values = []
    chosen_info_list = []
    
    # Loop over every eligible mixed-data base folder.
    for base_folder in mixed_base_folders:
        subdirs = [d for d in os.listdir(base_folder)
                   if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("n_disc_")]
        if not subdirs:
            print(f"Base folder {base_folder} has no n_disc subdirectories. Skipping.")
            continue

        selected_subdir = random.choice(subdirs)
        data_path = os.path.join(base_folder, selected_subdir)
        
        # Construct the training file path using the sample size.
        sample_size = args.n_samples
        train_file = f"train_n_samples_{sample_size}.csv"
        full_train_path = os.path.join(data_path, train_file)
        if not os.path.exists(full_train_path):
            print(f"Training file {full_train_path} does not exist in base folder {base_folder}. Skipping.")
            continue
        
        # Randomly select one seed.
        selected_seed = random.choice(seed_values)
        
        # Record the chosen combination for logging.
        info = (f"base_folder={os.path.basename(base_folder)} | "
                f"subdir={selected_subdir} | train_file={train_file} | seed={selected_seed}")
        chosen_info_list.append(info)
        
        metric_val = run_experiment(selected_seed, full_train_path,
                                    lam_h_val=lam_h_sample, lam_l1_val=lam_l1_sample)
        metric_values.append(metric_val)
    
    if not metric_values:
        trial.set_user_attr("chosen", "None")
        return 1e9  # Penalize trial if no valid experiments.
    
    trial.set_user_attr("chosen", chosen_info_list)
    trial.set_user_attr("metric_values", metric_values)
    
    return np.mean(metric_values)

def callback(study, trial):
    chosen_info = trial.user_attrs.get("chosen", "N/A")
    log_str = (f"Trial {trial.number}: chosen={chosen_info}, "
               f"trial_params={trial.params}, avg_normalised_SHD={trial.value} | "
               f"Best so far: params={study.best_params}, best_avg_normalised_SHD={study.best_value}")
    print(log_str)
    with open("hsearch_mixed_intermediate_best.txt", "a") as f:
        f.write(log_str + "\n")

# ------------------ Run Hyperparameter Optimization ------------------
if __name__ == "__main__":
    import optuna
    import random
    import os

    # PCX related imports
    import pcx as px
    import pcx.predictive_coding as pxc
    import pcx.nn as pxnn
    import pcx.functional as pxf
    import pcx.utils as pxu

    # JAX related imports
    import jax
    import jax.numpy as jnp
    import jax.numpy.linalg as jax_numpy_linalg
    import jax.scipy.linalg as jax_scipy_linalg 
    import jax.random as random

    # Deep learning frameworks
    import torch
    import optax

    # Causal discovery libraries
    import cdt
    import castle

    # Causal metrics
    from castle.metrics import MetricsDAG
    from castle.common import GraphDAG
    from causallearn.graph.SHD import SHD as SHD_causallearn

    # Custom modules
    from causal_model import Complete_Graph
    from causal_helpers import (
        enforce_dag, pick_source_target, is_dag_nx, MAE,
        compute_binary_adjacency, compute_h_reg, notears_dag_constraint,
        dagma_dag_constraint, load_adjacency_matrix, set_random_seed,
        plot_adjacency_matrices, load_graph
    )
    from causal_metrics import (
        compute_F1_directed, compute_F1_skeleton, compute_AUPRC,
        compute_AUROC, compute_cycle_F1, compute_SHD, compute_SID,
        compute_ancestor_AID, compute_TEE, compute_SHD_robust
    )

    # causal libraries
    
    os.environ["CASTLE_BACKEND"] = "pytorch"  # Prevents repeated backend initialization

    # Define the storage path for the persistent study.
    storage_path = "/share/amine.mcharrak/hsearch/mixed/optuna_study.db"
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    # Define the log file.
    intermediate_file = "hsearch_mixed_intermediate_best.txt"
    # Do not remove the log file to preserve previous entries.

    # Create a new study with persistent storage.
    study = optuna.create_study(
        study_name="my_mixed_study",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=2000, timeout=60*60*24*4, callbacks=[callback])
    
    print("Best hyperparameters: ", study.best_params)
    print("Best Normalized Final SHD: ", study.best_value)
    study.trials_dataframe().to_csv("optuna_study_results_mixed_normalised_shd.csv", index=False)