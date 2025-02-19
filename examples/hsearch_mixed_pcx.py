import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run experiments on a single folder.")
parser.add_argument("--path_name", type=str,
                    default="/share/amine.mcharrak/mixed_data_final/10ER40_linear_mixed_seed_1",
                    help="(For increase_n_disc) Base folder path (contains global files and n_disc_* subfolders).")
parser.add_argument("--n_samples", type=int, default=2000,
                    help="Number of training samples to use (to select the train_n_samples_<n>.csv file).")
parser.add_argument("--num_seeds", type=int, default=5,
                    help="Number of seeds to run experiments for.")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--exp_name", type=str, required=True,
                    choices=["mixed_confounding", "increase_n_disc"],
                    help="Name of the experiment.")
args = parser.parse_args()

# Set GPU-related environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
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
# Global hyperparameters (will be overridden in run_experiment)
lam_h = 5e0
lam_l1 = 1e-1

num_samples = args.n_samples
w_learning_rate = 1e-3
h_learning_rate = 1e-4
nm_epochs = 5000
batch_size = 256
every_n_epochs = nm_epochs // 5

# Define seed values.
seed_values = list(range(1, args.num_seeds + 1))

def run_experiment(seed, sample_size, data_file_path, base_folder=None, lam_h_val=None, lam_l1_val=None):
    global lam_h, lam_l1
    # Override global hyperparameters used in the loss.
    lam_h = lam_h_val
    lam_l1 = lam_l1_val

    set_random_seed(seed)
    print(f"🔄 Running experiment with seed {seed} on data file: {data_file_path}")
    
    global var_dict
    var_dict = {}  # Reset any global variable if needed

    if base_folder is not None:
        # For "increase_n_disc": load global matrices from base_folder
        adj_matrix = pd.read_csv(os.path.join(base_folder, 'adj_matrix.csv'), header=None)
        weighted_adj_matrix = pd.read_csv(os.path.join(base_folder, 'W_adj_matrix.csv'), header=None)
        # Here, data_file_path already points to the full training CSV file.
        train_file = data_file_path
    else:
        # For "mixed_confounding": all files are in data_path
        adj_matrix = pd.read_csv(os.path.join(data_file_path, 'adj_matrix.csv'), header=None)
        weighted_adj_matrix = pd.read_csv(os.path.join(data_file_path, 'W_adj_matrix.csv'), header=None)
        train_file = os.path.join(data_file_path, "train.csv")
    
    X_df = pd.read_csv(train_file, header=None)

    # Convert to numpy
    B_true = adj_matrix.values
    W_true = weighted_adj_matrix.values
    X = X_df.values  # This is needed for some models

    print(f"Original X_df shape: {X_df.shape}")

    # Subsample the data if sample_size is smaller than the full dataset
    if sample_size < X.shape[0]:
        np.random.seed(seed)  # Ensure same subsample for same seed
        sampled_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X = X[sampled_indices, :]  # Subsample X
        X_df = X_df.iloc[sampled_indices, :].reset_index(drop=True)  # Subsample X_df too

    print(f"Sampled X_df shape: {X_df.shape}")

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

        print("Energy: Starting computation")
        x_ = model(None)
        print("Energy: Got model output")
        
        W = model.get_W()
        # Dimensions of W
        d = W.shape[0]
        print(f"Energy: Got W (shape: {W.shape}) and d: {d}")

        # PC energy term
        pc_energy = jax.lax.pmean(model.energy(), axis_name="batch")
        print(f"Energy: PC energy term: {pc_energy}")

        # L1 regularization using adjacency matrix (scaled by Frobenius norm)
        l1_reg = jnp.sum(jnp.abs(W)) / (jnp.linalg.norm(W, ord='fro') + 1e-8)
        #l1_reg = jnp.sum(jnp.abs(W)) / d
        print(f"Energy: L1 reg term: {l1_reg}")

        # DAG constraint (stable logarithmic form)
        #h_reg = notears_dag_constraint(W)
        #h_reg = notears_dag_constraint(W)/ (jnp.sqrt(d) + 1e-8)
        #h_reg = notears_dag_constraint(W) / d  # with normalization

        #h_reg = dagma_dag_constraint(W)
        h_reg = dagma_dag_constraint(W) / (jnp.sqrt(d) + 1e-8)
        #h_reg = dagma_dag_constraint(W) / d  # with normalization
        print(f"Energy: DAG constraint term: {h_reg}")
            
        # Combined loss
        obj = pc_energy + lam_h * h_reg + lam_l1 * l1_reg
        print(f"Energy: Final objective: {obj}")

        # Ensure obj is a scalar, not a (1,) array because JAX's grad and value_and_grad functions are designed
        # to compute gradients of *scalar-output functions*
        obj = obj.squeeze()  # explicitly converte the (1,) array (obj) to a scalar of shape ()  
        
        return obj, pc_energy, h_reg, l1_reg, x_

    @pxf.jit(static_argnums=0)
    def train_on_batch(T: int, x: jax.Array, *, model: Complete_Graph, optim_w: pxu.Optim, optim_h: pxu.Optim):
        print("1. Starting train_on_batch")  

        model.train()
        print("2. Model set to train mode")

        model.freeze_nodes(freeze=True)
        print("3. Nodes frozen")

        # init step
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            print("4. Doing forward for initialization")
            forward(x, model=model)
            print("5. After forward for initialization")


        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            print("6. Before computing gradients")
            (obj, (pc_energy, h_reg, l1_reg, x_)), g = pxf.value_and_grad(
                pxu.M(pxnn.LayerParam).to([False, True]), 
                has_aux=True
            )(energy)(model=model) # pxf.value_and_grad returns a tuple structured as ((value, aux), grad), not as six separate outputs.
            
            print("7. After computing gradients")
            #print("Gradient structure:", g)

            print("8. Before zeroing out the diagonal gradients")
            # Zero out the diagonal gradients using jax.numpy.fill_diagonal
            weight_grads = g["model"].layers[0].nn.weight.get()
            weight_grads = jax.numpy.fill_diagonal(weight_grads, 0.0, inplace=False)
            # print the grad values using the syntax jax.debug.print("🤯 {x} 🤯", x=x)
            #jax.debug.print("{weight_grads}", weight_grads=weight_grads)
            g["model"].layers[0].nn.weight.set(weight_grads)
            print("9. After zeroing out the diagonal gradients")

            
        print("10. Before optimizer step")
        optim_w.step(model, g["model"])
        #optim_w.step(model, g["model"], scale_by=1.0/x.shape[0])
        print("11. After optimizer step")

        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            print("12. Before final forward")
            forward(None, model=model)
            e_avg_per_sample = model.energy()
            print("13. After final forward")

        model.freeze_nodes(freeze=False)
        print("14. Nodes unfrozen")

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
    
    sid_metrics = []
    import timeit
    start_time = timeit.default_timer()
    for epoch in tqdm(range(nm_epochs)):
        W, ep_pc, ep_l1, ep_h, ep_obj = train(dl, T=1, model=model, optim_w=optim_w, optim_h=optim_h)
        if (epoch + 1) % every_n_epochs == 0 or epoch == 0:
            B_est = compute_binary_adjacency(np.array(W))
            # For SID, assume compute_SID returns a tuple (normalized, absolute)
            sid_norm, _ = compute_SID((enforce_dag(B_true) > 0).astype(np.int8), enforce_dag(B_est))
            sid_metrics.append(sid_norm)
            tqdm.write(f"Epoch {epoch+1}: SID_normalized = {sid_norm:.4f}")
    end_time = timeit.default_timer()
    print(f"Average epoch time: {(end_time - start_time) / nm_epochs:.4f} seconds")
    
    # After training, compute final SID_normalized.
    W_final = model.get_W()
    B_final = enforce_dag(compute_binary_adjacency(np.array(W_final)))
    final_sid_norm, _ = compute_SID((enforce_dag(B_true) > 0).astype(np.int8), B_final)
    print(f"Final SID_normalized: {final_sid_norm}")
    return final_sid_norm

# ------------------ Define the Optuna Objective Function ------------------
def objective(trial):
    # Sample hyperparameters.
    lam_h_sample = trial.suggest_float("lam_h", 1e-5, 1e3, log=True)
    lam_l1_sample = trial.suggest_float("lam_l1", 1e-5, 1e3, log=True)
    
    # List of eligible base folders.
    base_folders = [
        "/share/amine.mcharrak/mixed_data_final/10ER40_linear_mixed_seed_1",
        "/share/amine.mcharrak/mixed_data_final/10SF40_linear_mixed_seed_1",
        "/share/amine.mcharrak/mixed_data_final/10WS40_linear_mixed_seed_1"
    ]
    selected_base_folder = trial.suggest_categorical("base_folder", base_folders)
    
    # List subdirectories starting with "n_disc_" in the selected base folder.
    subdirs = [d for d in os.listdir(selected_base_folder)
               if os.path.isdir(os.path.join(selected_base_folder, d)) and d.startswith("n_disc_")]
    if not subdirs:
        trial.set_user_attr("sid", None)
        return 1e9
    selected_subdir = trial.suggest_categorical("n_disc_subdir", subdirs)
    data_path = os.path.join(selected_base_folder, selected_subdir)
    
    # List available training files in the selected subdirectory.
    train_files = [f for f in os.listdir(data_path)
                   if f.startswith("train_n_samples_") and f.endswith(".csv")]
    if not train_files:
        trial.set_user_attr("sid", None)
        return 1e9
    selected_train_file = trial.suggest_categorical("train_file", train_files)
    full_data_path = os.path.join(data_path, selected_train_file)
    
    # Randomly select a seed.
    selected_seed = trial.suggest_categorical("seed", [str(s) for s in seed_values])
    selected_seed = int(selected_seed)
    
    # Pass the sampled hyperparameters to run_experiment.
    sid_val = run_experiment(selected_seed, args.n_samples, full_data_path,
                             base_folder=selected_base_folder,
                             lam_h_val=lam_h_sample,
                             lam_l1_val=lam_l1_sample)
    if sid_val is None:
        trial.set_user_attr("sid", None)
        return 1e9
    trial.set_user_attr("sid", sid_val)
    return sid_val

def callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value}")
    print("Current best hyperparameters:", study.best_params)
    with open("hsearch_mixed_intermediate_best.txt", "a") as f:
        f.write(f"Trial {trial.number}: best={study.best_params}, value={study.best_value}\n")

# ------------------ Run Hyperparameter Optimization ------------------
if __name__ == "__main__":

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
    from causal_helpers import enforce_dag, pick_source_target
    from causal_helpers import is_dag_nx, MAE, compute_binary_adjacency, compute_h_reg, notears_dag_constraint, dagma_dag_constraint
    from causal_helpers import load_adjacency_matrix, set_random_seed, plot_adjacency_matrices
    from causal_helpers import load_graph, load_adjacency_matrix
    from causal_metrics import compute_F1_directed, compute_F1_skeleton, compute_AUPRC, compute_AUROC, compute_cycle_F1
    from causal_metrics import compute_SHD, compute_SID, compute_ancestor_AID, compute_TEE

    # causal libraries
    
    # lingam
    import lingam

    os.environ["CASTLE_BACKEND"] = "pytorch"  # Prevents repeated backend initialization
    import cdt, castle

    from castle.algorithms import Notears, ICALiNGAM, PC, DirectLiNGAM
    from castle.algorithms import DirectLiNGAM, GOLEM
    from castle.algorithms.ges.ges import GES
    from castle.algorithms import PC

    # causal metrics
    from cdt.metrics import precision_recall, SHD, SID
    from castle.metrics import MetricsDAG
    from castle.common import GraphDAG
    from causallearn.graph.SHD import SHD as SHD_causallearn
    
    # Set persistent storage location.
    storage_path = "/share/amine.mcharrak/hsearch/mixed/optuna_study.db"
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    
    study = optuna.create_study(
        study_name="my_increase_n_disc_study",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize"
    )
    
    study.optimize(objective, n_trials=1000, timeout=60*60*24*2, callbacks=[callback])
    
    print("Best hyperparameters: ", study.best_params)
    print("Best SID_normalized: ", study.best_value)
    study.trials_dataframe().to_csv("optuna_study_results_increase_n_disc.csv", index=False)