import os
import argparse
# Argument Parsing (before importing GPU-dependent libraries)
parser = argparse.ArgumentParser(description="Run structured experiments.")

parser.add_argument("--gpu", type=int, default=0,
                    help="GPU index to use")
parser.add_argument("--num_seeds", type=int, default=10,
                    help="Number of seeds for experiments")
args = parser.parse_args()

# Set GPU-related environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["JAX_PLATFORM_NAME"] = "cuda"  # Force JAX to use CUDAimport jax.numpy as jnp


# Your experiment paths and parameters.
data_dir = "/share/amine.mcharrak/cyclic_data_final"
num_seeds = args.num_seeds
seed_values = list(range(1, num_seeds + 1))

experiment_paths = {
        "ER": [
            "10ER10_linear_cyclic_GAUSS-EV_seed_1",
            "10ER20_linear_cyclic_GAUSS-EV_seed_1",
            "10ER30_linear_cyclic_GAUSS-EV_seed_1",
            "10ER41_linear_cyclic_GAUSS-EV_seed_1",
            "15ER16_linear_cyclic_GAUSS-EV_seed_1",
            "15ER31_linear_cyclic_GAUSS-EV_seed_1",
            "15ER46_linear_cyclic_GAUSS-EV_seed_1",
            "15ER62_linear_cyclic_GAUSS-EV_seed_1",
            "20ER22_linear_cyclic_GAUSS-EV_seed_1",
            "20ER39_linear_cyclic_GAUSS-EV_seed_1",
            "20ER59_linear_cyclic_GAUSS-EV_seed_1",
            "20ER81_linear_cyclic_GAUSS-EV_seed_1"
        ],
        "SF": [
            "10SF10_linear_cyclic_GAUSS-EV_seed_1",
            "10SF20_linear_cyclic_GAUSS-EV_seed_1",
            "10SF30_linear_cyclic_GAUSS-EV_seed_1",
            "10SF41_linear_cyclic_GAUSS-EV_seed_1",
            "15SF16_linear_cyclic_GAUSS-EV_seed_1",
            "15SF28_linear_cyclic_GAUSS-EV_seed_1",
            "15SF45_linear_cyclic_GAUSS-EV_seed_1",
            "15SF61_linear_cyclic_GAUSS-EV_seed_1",
            "20SF50_linear_cyclic_GAUSS-EV_seed_1",
            "20SF60_linear_cyclic_GAUSS-EV_seed_1",
            "20SF65_linear_cyclic_GAUSS-EV_seed_1",
            "20SF79_linear_cyclic_GAUSS-EV_seed_1",
            "20SF97_linear_cyclic_GAUSS-EV_seed_1"
        ],
        "NWS": [
            "10NWS9_linear_cyclic_GAUSS-EV_seed_1",
            "10NWS18_linear_cyclic_GAUSS-EV_seed_1",
            "10NWS25_linear_cyclic_GAUSS-EV_seed_1",
            "10NWS34_linear_cyclic_GAUSS-EV_seed_1",
            "15NWS17_linear_cyclic_GAUSS-EV_seed_1", 
            "15NWS31_linear_cyclic_GAUSS-EV_seed_1",
            "15NWS44_linear_cyclic_GAUSS-EV_seed_1",
            "15NWS48_linear_cyclic_GAUSS-EV_seed_1", 
            "15NWS60_linear_cyclic_GAUSS-EV_seed_1", 
            "20NWS19_linear_cyclic_GAUSS-EV_seed_1", 
            "20NWS42_linear_cyclic_GAUSS-EV_seed_1", 
            "20NWS60_linear_cyclic_GAUSS-EV_seed_1", 
            "20NWS92_linear_cyclic_GAUSS-EV_seed_1"
        ],
    }


import timeit
import numpy as np
import optuna
import torch
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

# pcx, jax, optax and your own modules
# pcx
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

# 3rd party
import torch
import jax
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as jax_numpy_linalg # for expm()
import jax.scipy.linalg as jax_scipy_linalg # for slogdet()
import jax.random as random
import optax

# own
import causal_helpers
from causal_model import Complete_Graph
from causal_helpers import (
    is_dag_nx, MAE, compute_binary_adjacency, compute_h_reg, 
    notears_dag_constraint, dagma_dag_constraint,
    simulate_dag, simulate_parameter, simulate_linear_sem, simulate_linear_sem_cyclic,
    load_adjacency_matrix, set_random_seed, plot_adjacency_matrices,
    load_graph
)
from causal_metrics import (
    compute_F1_directed, compute_F1_skeleton, compute_AUPRC, compute_AUROC, 
    compute_model_fit, compute_cycle_F1, compute_cycle_SHD, compute_cycle_KLD, compute_CSS
)

# DAG search utilities
from dglearn.dglearn.dg.adjacency_structure import AdjacencyStucture
from dglearn.dglearn.dg.graph_equivalence_search import GraphEquivalenceSearch
from dglearn.dglearn.dg.converter import binary2array

import optax

# =============================================================================
# Global hyperparameters (will be overridden in run_experiment)
lam_h = 5e0      # effective value depends on d, does not change during training
lam_l1 = 1e-1    # effective value depends on Frobenius norm of W at any time

# Other parameters (you can leave these fixed)
num_samples = 2000
w_learning_rate = 1e-3
h_learning_rate = 1e-4
nm_epochs = 5000
batch_size = 256
every_n_epochs = nm_epochs // 5

# Function to load adjacency matrices and data
def load_data(base_path, num_samples):
    """
    Load adjacency matrices and data from the given base path and sample size.
    """
    adj_matrix = pd.read_csv(os.path.join(base_path, 'adj_matrix.csv'), header=None)
    weighted_adj_matrix = pd.read_csv(os.path.join(base_path, 'W_adj_matrix.csv'), header=None)
    prec_matrix = pd.read_csv(os.path.join(base_path, 'prec_matrix.csv'), header=None)

    data_path = os.path.join(base_path, f"n_samples_{num_samples}", "train.csv")
    data = pd.read_csv(data_path, header=None)

    return adj_matrix, weighted_adj_matrix, prec_matrix, data

def safe_compute(func, *args, default_value=None, **kwargs):
    """
    Wrapper to safely compute a function and handle exceptions.
    Returns a default value (None by default) if an error occurs.
    Supports both positional and keyword arguments.
    """
    try:
        value = func(*args, **kwargs)  # Pass keyword arguments correctly
        
        # If function returns None, replace with default_value
        if value is None:
            return default_value

        # If function returns a list/tuple, check each element for validity
        if isinstance(value, (list, tuple)):  
            return [v if isinstance(v, (int, float, np.number)) and np.isfinite(v) else default_value for v in value]

        # If function returns a scalar, check if it's a valid number
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            return value
        
        # If none of the above conditions matched, return default_value
        return default_value

    except Exception as e:
        print(f"⚠️ Error computing {func.__name__}: {e}")
        return default_value

# =============================================================================
# Define run_experiment to run training on one dataset with given hyperparameters.
def run_experiment(seed, base_path, lam_h_val, lam_l1_val):
    global lam_h, lam_l1
    # Override global hyperparameters used in the loss.
    lam_h = lam_h_val
    lam_l1 = lam_l1_val

    # Set random seed.
    #set_random_seed(seed) # don't set seed here, as it makes choosing graphs and seeds deterministic
    print(f"Running experiment on {base_path} with seed {seed}, lam_h={lam_h_val}, lam_l1={lam_l1_val}")
    
    # Load data and true matrices.
    adj_matrix, weighted_adj_matrix, prec_matrix, data = load_data(base_path, num_samples)
    B_true = adj_matrix.values
    X = data.values
    W_true = weighted_adj_matrix.values
    
    # Build true graph and equivalence class.
    G_true = nx.DiGraph(B_true)
    edges_list = list(G_true.edges())
    true_graph = AdjacencyStucture(n_vars=data.shape[1], edge_list=edges_list)
    search = GraphEquivalenceSearch(true_graph)
    search.search_dfs()
    B_true_EC = [binary2array(bstr) for bstr in search.visited_graphs]
    print(f"Equivalence class size: {len(B_true_EC)}")
    
    # Determine which nodes are continuous.
    is_cont_node = [True if data[col].nunique() > 2 else False for col in data.columns]
    print("Continuous variables:", is_cont_node)
    
    # Initialize the model.
    n_nodes = X.shape[1]
    model = Complete_Graph(input_dim=1, n_nodes=n_nodes, has_bias=True, is_cont_node=is_cont_node, seed=seed)
    
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
    
    # Define training and evaluation functions (using your existing code).
    @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=0)
    def forward(x, *, model: Complete_Graph):
        return model(x)
    
    @pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), 
             out_axes=(None, None, None, None, 0), axis_name="batch")
    def energy(*, model: Complete_Graph):
        # Get model output (unused) and W.
        _ = model(None)
        W = model.get_W()
        d = W.shape[0]
        pc_energy = jax.lax.pmean(model.energy(), axis_name="batch")
        l1_reg = jnp.sum(jnp.abs(W)) / (jnp.linalg.norm(W, ord='fro') + 1e-8)
        h_reg = dagma_dag_constraint(W) / (jnp.sqrt(d) + 1e-8)
        obj = pc_energy + lam_h * h_reg + lam_l1 * l1_reg
        return obj.squeeze(), pc_energy, h_reg, l1_reg, None

    @pxf.jit(static_argnums=0)
    def train_on_batch(T: int, x: jax.Array, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):
        model.train()
        model.freeze_nodes(freeze=True)
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(x, model=model)
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            (obj, (pc_energy, h_reg, l1_reg, _)), g = pxf.value_and_grad(
                pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True
            )(energy)(model=model)
            # Zero out diagonal gradients.
            weight_grads = g["model"].layers[0].nn.weight.get()
            weight_grads = jax.numpy.fill_diagonal(weight_grads, 0.0, inplace=False)
            g["model"].layers[0].nn.weight.set(weight_grads)
        optim_w.step(model, g["model"])
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            forward(None, model=model)
            e_avg = model.energy()
        model.freeze_nodes(freeze=False)
        return pc_energy, l1_reg, h_reg, obj

    def train(dl, T, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):
        batch_pc_energies = []
        batch_l1_regs = []
        batch_h_regs = []
        batch_objs = []
        for batch in dl:
            pc_energy, l1_reg, h_reg, obj = train_on_batch(T, batch, model=model, optim_w=optim_w, optim_h=optim_h)
            batch_pc_energies.append(pc_energy)
            batch_l1_regs.append(l1_reg)
            batch_h_regs.append(h_reg)
            batch_objs.append(obj)
        W = model.get_W()
        epoch_pc_energy = jnp.mean(jnp.array(batch_pc_energies))
        epoch_l1_reg = jnp.mean(jnp.array(batch_l1_regs))
        epoch_h_reg = jnp.mean(jnp.array(batch_h_regs))
        epoch_obj = jnp.mean(jnp.array(batch_objs))
        return W, epoch_pc_energy, epoch_l1_reg, epoch_h_reg, epoch_obj

    # Initialize optimizers.
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, model.n_nodes.get())), model=model)
        optim_h = pxu.Optim(lambda: optax.sgd(h_learning_rate))
        optim_w = pxu.Optim(lambda: optax.adamw(w_learning_rate, nesterov=True), pxu.M(pxnn.LayerParam)(model))
    
    # Training loop.
    css_metrics = []
    start_time = timeit.default_timer()
    for epoch in tqdm(range(nm_epochs)):
        W, ep_pc, ep_l1, ep_h, ep_obj = train(dl, T=1, model=model, optim_w=optim_w, optim_h=optim_h)
        if (epoch + 1) % every_n_epochs == 0 or epoch == 0:
            B_est = compute_binary_adjacency(np.array(W))
            shd_cyc = compute_cycle_SHD(B_true_EC, B_est)
            f1_cyc = compute_cycle_F1(B_true, B_est)
            css_val = safe_compute(compute_CSS, shd_cyc, f1_cyc, epsilon=1e-8, default_value=1e9)
            # Normalize CSS by number of true edges
            true_edge_count = np.sum(B_true)
            normalized_css = css_val / true_edge_count if true_edge_count > 0 else css_val
            css_metrics.append(normalized_css)
            tqdm.write(f"Epoch {epoch+1}: CSS = {css_val:.4f}, Normalized CSS = {normalized_css:.4f}")
    end_time = timeit.default_timer()
    print(f"Average epoch time: {(end_time - start_time) / nm_epochs:.4f} seconds")
    
    # After training, compute final CSS.
    W_final = model.get_W()
    B_final = compute_binary_adjacency(np.array(W_final))
    final_shd_cyc = compute_cycle_SHD(B_true_EC, B_final)
    final_f1_cyc = compute_cycle_F1(B_true, B_final)
    final_css = safe_compute(compute_CSS, final_shd_cyc, final_f1_cyc, epsilon=1e-8, default_value=1e9)
    # Normalize by the number of true edges.
    true_edge_count = np.sum(B_true)
    normalized_final_css = final_css / true_edge_count if true_edge_count > 0 else final_css
    print(f"Final CSS: {final_css}")
    print(f"Normalized Final CSS (per true edge): {normalized_final_css}")
    return normalized_final_css

# =============================================================================

def objective(trial):
    import random

    # Sample only the hyperparameters.
    lam_h_sample = trial.suggest_float("lam_h", 1e-5, 1e3, log=True)
    lam_l1_sample = trial.suggest_float("lam_l1", 1e-5, 1e3, log=True)
    
    # Combine all eligible (graph_type, folder) pairs.
    all_pairs = []
    for gtype, folder_list in experiment_paths.items():
        for folder in folder_list:
            all_pairs.append((gtype, folder))
    if not all_pairs:
        raise ValueError("No graph-folder pairs found in experiment_paths.")
    
    # Set how many random (graph, seed) combinations to average over.
    num_combinations = 6
    if len(all_pairs) < num_combinations:
        num_combinations = len(all_pairs)
    
    chosen = []  # To store chosen (graph, seed) for this trial.
    metric_values = []
    
    # Sample without replacement.
    selected_pairs = random.sample(all_pairs, num_combinations)
    for selected_pair in selected_pairs:
        gtype, folder = selected_pair
        combo_info = f"{gtype}|{folder}"
        dataset_path = os.path.join(data_dir, folder)
        selected_seed = random.choice(seed_values)
        combo_info += f", seed={selected_seed}"
        chosen.append(combo_info)
        
        metric_val = run_experiment(selected_seed, dataset_path,
                                    lam_h_val=lam_h_sample, lam_l1_val=lam_l1_sample)
        metric_values.append(metric_val)
    
    trial.set_user_attr("chosen", chosen)
    trial.set_user_attr("metric_values", metric_values)
    
    return np.mean(metric_values)

def callback(study, trial):
    # Retrieve the chosen (graph, seed) pairs for this trial.
    chosen_info = trial.user_attrs.get("chosen", "N/A")
    log_str = (f"Trial {trial.number}: chosen={chosen_info}, "
               f"trial_params={trial.params}, avg_normalized_CSS={trial.value} | "
               f"Best so far: params={study.best_params}, best_avg_normalized_CSS={study.best_value}")
    print(log_str)
    with open("hsearch_cyclic_intermediate_best.txt", "a") as f:
        f.write(log_str + "\n")

# =============================================================================
# Main script to run optimization.
if __name__ == "__main__":
    import optuna
    import random
    import os

    # Remove previous study database and log file if they exist.
    storage_path = "/share/amine.mcharrak/hsearch/cyclic/optuna_study.db"
    if os.path.exists(storage_path):
        os.remove(storage_path)
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    intermediate_file = "hsearch_cyclic_intermediate_best.txt"
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)
    
    # Create a new study with persistent storage.
    study = optuna.create_study(
        study_name="my_study",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=False,
        direction="minimize"
    )
    
    # Run optimization.
    study.optimize(objective, n_trials=1000, timeout=60*60*24*2, callbacks=[callback])
    
    print("Best hyperparameters: ", study.best_params)
    print("Best Normalized Final CSS (per true edge): ", study.best_value)
    study.trials_dataframe().to_csv("optuna_study_results_cyclic_normalised_css.csv", index=False)
