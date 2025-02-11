import multiprocessing as mp
mp.set_start_method("fork", force=True)
#mp.set_start_method("spawn", force=True)
#mp.set_start_method("forkserver", force=True)

import os
import argparse
import sys

# Argument Parsing (before importing GPU-dependent libraries)
parser = argparse.ArgumentParser(description="Run structured experiments.")
parser.add_argument("--exp_name", type=str, required=True, choices=["increase_k", "increase_d", "increase_n"],
                    help="Specify the type of experiment (increase_k, increase_d, increase_n).")
parser.add_argument("--graph_type", type=str, required=True, choices=["ER", "SF", "NWS"],
                    help="Specify the graph type (ER, SF, NWS).")
parser.add_argument("--num_seeds", type=int, required=True, help="Number of seeds to run experiments for.")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")

args = parser.parse_args()

# Set GPU before importing any ML/DL libraries
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  

import numpy as np
import pandas as pd
import random
import json

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import timeit

###################################################

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

# Function to run the experiment
def run_experiment(seed, base_path, num_samples):
    """Runs the experiment for a given seed and returns results as a dictionary."""
    
    set_random_seed(seed)
    print(f"🔄 Running experiment with seed {seed}")

    # Load adjacency matrices, data, and precision matrix
    adj_matrix, weighted_adj_matrix, prec_matrix, data = load_data(base_path, num_samples)

    # Get the number of variables
    n_vars = data.shape[1]

    B_true = adj_matrix.values
    X = data.values
    W_true = weighted_adj_matrix.values

    # Get edge list from B_true using networkx
    G_true = nx.DiGraph(B_true)
    edges_list = list(G_true.edges())

    # Compute performance metric: SHD
    true_graph = AdjacencyStucture(n_vars=n_vars, edge_list=edges_list)
    search = GraphEquivalenceSearch(true_graph)
    search.search_dfs()

    B_true_EC = [binary2array(bstr) for bstr in search.visited_graphs]
    print(f"Size of equivalence class of true graph for {base_path}: {len(B_true_EC)}")

    # Check if G_true is cyclic by listing all cycles
    print(f"Number of cycles in G_true for {base_path}: {len(list(nx.simple_cycles(G_true)))}\n")

    # %%
    data.head()
    # show unique values in each column
    print(data.nunique())
    # Determine if each variable is continuous or discrete based on the number of unique values
    is_cont_node = np.array([True if data[col].nunique() > 2 else False for col in data.columns])
    is_cont_node = is_cont_node.tolist()
    print(is_cont_node)

    # print how many non-zero entries are in the true DAG
    print(f"Number of non-zero entries in the true DAG: {np.count_nonzero(B_true)}")

    # Usage
    input_dim = 1
    n_samples = X.shape[0]
    n_nodes = X.shape[1]
    #model = Complete_Graph(input_dim, n_nodes, has_bias=False, is_cont_node=is_cont_node, seed=seed)
    model = Complete_Graph(input_dim, n_nodes, has_bias=True, is_cont_node=is_cont_node, seed=seed)
    # Get weighted adjacency matrix
    W = model.get_W()
    print("This is the weighted adjacency matrix:\n", W)
    print()
    print("The shape of the weighted adjacency matrix is: ", W.shape)
    print()
    #print(model)
    print()

    # Freezing all nodes
    print("Freezing all nodes...")
    model.freeze_nodes(freeze=True)
    print()

    # Check if all nodes are frozen after freezing
    print("After freezing, are all nodes frozen?:", model.are_vodes_frozen())
    print()

    # Unfreezing all nodes
    print("Unfreezing all nodes...")
    model.freeze_nodes(freeze=False)
    print()

    # Check if all nodes are frozen after unfreezing
    print("After unfreezing, are all nodes frozen?:", model.are_vodes_frozen())

    model.is_cont_node

    # %%
    # TODO: make the below params global or input to the functions in which it is used.
    w_learning_rate = 1e-3
    h_learning_rate = 1e-4
    T = 1

    nm_epochs = 5000 # not much happens after 5000 epochs
    every_n_epochs = nm_epochs // 20 # Print every 5% of the epochs
    batch_size = 256

    # NOTE: small lam_l1 (1e-5) and larger lam_h (1e1) seem to work well
    # NOTE: for any graph type, the best way to select lam_h is such that one increases it until the graph becomes acyclic, a value right before that is the best value
    # NOTE: larger lam_h (1e1) gives better cyclic structure score (CSS) but worse fit metric scores (log-likelihood, KL divergence, etc.)
    # NOTE: smaller lam_h (5e-4) gives better fit metric scores (log-likelihood, KL divergence, etc.) but worse cyclic structure score (CSS)
    
    lam_h = 5e0 # effective value depends on d, does not change during training
    lam_l1 = 1e0 # effective value depends on Frob Norm of W at any time, thus changes during training

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

        """
        # The following code might not be needed as we are keeping the vodes frozen at all times
        # Reinitialize the optimizer state between different batches
        optim_h.init(pxu.M(pxc.VodeParam)(model))

        for _ in range(T):
            with pxu.step(model, clear_params=pxc.VodeParam.Cache):
                (e, x_), g = pxf.value_and_grad(
                    pxu.M(pxu.m(pxc.VodeParam).has_not(frozen=True), [False, True]),
                    has_aux=True
                )(energy)(model=model)
            optim_h.step(model, g["model"], True)
        """

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


    # Create the custom dataset
    dataset = CustomDataset(X)

    # Create the custom dataset with standardized data
    #dataset_std = CustomDataset(X_std)

    # Create the dataloader
    dl = TorchDataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, prefetch_factor=None, persistent_workers=False)
    ######## OR ########
    #dl = TorchDataloader(dataset_std, batch_size=batch_size, shuffle=True)

    # %%
    # Initialize the model and optimizers
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(jnp.zeros((batch_size, model.n_nodes.get())), model=model)
        optim_h = pxu.Optim(lambda: optax.sgd(h_learning_rate))

        """
        optim_w = pxu.Optim(
        optax.chain(
            optax.clip_by_global_norm(clip_value),  # Clip gradients by global norm
            optax.sgd(w_learning_rate)  # Apply SGD optimizer
        ),
        pxu.M(pxnn.LayerParam)(model)  # Masking the parameters of the model
    )
        """
        #optim_w = pxu.Optim(lambda: optax.adam(w_learning_rate), pxu.M(pxnn.LayerParam)(model)) # adam without weight decay
        optim_w = pxu.Optim(lambda: optax.adamw(w_learning_rate, nesterov=True), pxu.M(pxnn.LayerParam)(model)) # adam with weight decay

    # %%
    # Initialize lists to store differences and energies
    MAEs = []
    SHDs = []
    SHDs_cyclic = []
    F1s_cycles = []
    F1s = []
    pc_energies = []
    l1_regs = []
    h_regs = []
    objs = []

    # Calculate the initial MAE, SID, and SHD

    W_init = model.get_W()
    B_init = compute_binary_adjacency(W_init)

    MAE_init = MAE(W_true, W_init)
    print(f"Start difference (cont.) between W_true and W_init: {MAE_init:.4f}")

    SHD_init = SHD(B_true, B_init, double_for_anticausal=False)
    print(f"Start SHD between B_true and B_init: {SHD_init:.4f}")

    SHD_cyclic_init = compute_cycle_SHD(B_true_EC, B_init)
    print(f"Start SHD (cyclic) between B_true and B_init: {SHD_cyclic_init:.4f}")

    F1_init = compute_F1_directed(B_true, B_init)
    print(f"Start F1 between B_true and B_init: {F1_init:.4f}")

    cycle_f1_init = compute_cycle_F1(B_true, B_init)
    print(f"Start cycle accuracy between B_true and B_init: {cycle_f1_init:.4f}")

    # print the values of the diagonal of the initial W
    print("The diagonal of the initial W: ", jnp.diag(W_init))

    # Start timing
    start_time = timeit.default_timer()

    # Training loop
    with tqdm(range(nm_epochs), position=0, leave=True) as pbar:
        for epoch in pbar:
            # Train for one epoch using the dataloader
            W, epoch_pc_energy, epoch_l1_reg, epoch_h_reg, epoch_obj = train(dl, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
            
            # Extract the weighted adjacency matrix W and compute the binary adjacency matrix B
            W = np.array(W)
            B = compute_binary_adjacency(W)

            # Compute metrics every 100 epochs
            if (epoch + 1) % every_n_epochs == 0 or epoch == 0:
                MAEs.append(float(MAE(W_true, W)))
                SHDs.append(float(SHD(B_true, compute_binary_adjacency(W), double_for_anticausal=False)))
                SHDs_cyclic.append(float(compute_cycle_SHD(B_true_EC, compute_binary_adjacency(W))))
                F1s.append(float(compute_F1_directed(B_true, B)))
                F1s_cycles.append(float(compute_cycle_F1(B_true, B)))
                pc_energies.append(float(epoch_pc_energy))
                l1_regs.append(float(epoch_l1_reg))
                epoch_h_reg_raw = compute_h_reg(W)
                h_regs.append(float(epoch_h_reg_raw))
                objs.append(float(epoch_obj))

                # Update progress bar with the current status
                pbar.set_description(f"MAE: {MAEs[-1]:.4f}, Cycle F1: {F1s_cycles[-1]:.4f}, F1: {F1s[-1]:.4f}, SHD: {SHDs[-1]:.4f}, Cycle SHD: {SHDs_cyclic[-1]:.4f} || PC Energy: {pc_energies[-1]:.4f}, L1 Reg: {l1_regs[-1]:.4f}, H Reg: {h_regs[-1]:.4f}, Obj: {objs[-1]:.4f}")

    # End timing
    end_time = timeit.default_timer()

    # Print the average time per epoch
    average_time_per_epoch = (end_time - start_time) / nm_epochs
    print(f"An epoch (with compiling and testing) took on average: {average_time_per_epoch:.4f} seconds")
    # print the values of the diagonal of the final W
    print("The diagonal of the final W: ", jnp.diag(model.get_W()))


    # print in big that training is done
    print("\n\n ###########################  Training is done  ########################### \n\n")

    # %%
    # Create a file name string for the hyperparameters
    exp_name = f"bs_{batch_size}_lrw_{w_learning_rate}_lrh_{h_learning_rate}_lamh_{lam_h}_laml1_{lam_l1}_epochs_{nm_epochs}_seed_{seed}"
    print("Name of the experiment: ", exp_name)    

    # Create subdirectory in linear folder with the name stored in exp_name
    last_level = os.path.basename(os.path.normpath(base_path))
    save_path = os.path.join('plots/linear_cyclic', last_level, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Reset to default style and set seaborn style
    plt.style.use('default')
    sns.set_style("whitegrid")

    # Update matplotlib parameters
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    # Create a figure and subplots using GridSpec with 3x3 layout
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

    # Adjust layout to make room for titles and subtitles
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

    # Create axes
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    # Updated plot configurations
    plot_configs = [
        {'metric': MAEs, 'title': 'MAE', 'ylabel': 'MAE', 'color': '#2ecc71', 'ax': axes[0]},
        {'metric': SHDs, 'title': 'SHD', 'ylabel': 'SHD', 'color': '#e74c3c', 'ax': axes[1]},
        {'metric': SHDs_cyclic, 'title': 'SHD Cyclic', 'ylabel': 'SHD Cyclic', 'color': '#8e44ad', 'ax': axes[2]},
        {'metric': F1s, 'title': 'F1 Score', 'ylabel': 'F1 Structure', 'color': '#3498db', 'ax': axes[3]},
        {'metric': l1_regs, 'title': 'L1 Regularization', 'ylabel': 'L1', 'color': '#f1c40f', 'ax': axes[4]},
        {'metric': h_regs, 'title': 'DAG Constraint', 'ylabel': 'h(W)', 'color': '#e67e22', 'ax': axes[5]},
        {'metric': pc_energies, 'title': 'PC Energy', 'ylabel': 'PC Energy', 'color': '#9b59b6', 'ax': axes[6]},
        {'metric': objs, 'title': 'Total Objective', 'ylabel': 'Total Loss', 'color': '#1abc9c', 'ax': axes[7]},
        {'metric': F1s_cycles, 'title': 'F1 Cyclic', 'ylabel': 'F1 Cyclic', 'color': '#16a085', 'ax': axes[8]}
    ]

    # Create all subplots
    for config in plot_configs:
        ax = config['ax']

        epochs = range(0, len(config['metric']) * every_n_epochs, every_n_epochs)
        
        # Determine if we should use log scale and/or scaling factor
        use_log_scale = config['title'] in ['PC Energy', 'Total Objective']
        #scale_factor = 1e4 if config['title'] == 'DAG Constraint' else 1
        scale_factor = 1 if config['title'] == 'DAG Constraint' else 1
        
        # Apply scaling and/or log transform to the metric
        metric_values = np.array(config['metric'])
        if use_log_scale:
            # Add small constant to avoid log(0)
            metric_values = np.log10(np.abs(metric_values) + 1e-10)
        metric_values = metric_values * scale_factor

        # Plot raw data
        ax.plot(epochs, metric_values, alpha=0.3, color=config['color'], label='Raw')

        # Calculate rolling average
        window_size = max(1, len(metric_values) // 50)  # Dynamic window size based on number of epochs
        if len(metric_values) > window_size:
            rolling_mean = np.convolve(metric_values, np.ones(window_size) / window_size, mode='valid')
            ax.plot(epochs[window_size - 1:], rolling_mean, color=config['color'], linewidth=2, label='Rolling Avg')

        # Customize subplot
        ax.set_xlabel('Epoch', labelpad=10)
        # Adjust ylabel based on transformations
        ylabel = config['ylabel']
        if use_log_scale:
            ylabel = f'log10({ylabel})'
        if scale_factor != 1:
            ylabel = f'{ylabel} (×{int(scale_factor)})'
        ax.set_ylabel(ylabel, labelpad=10)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend for Total Objective plot
        if config['title'] == 'Total Objective':
            ax.legend(loc='upper right')

        # Add note about scaling if applicable
        if use_log_scale or scale_factor != 1:
            transform_text = []
            if use_log_scale:
                transform_text.append('log scale')
            if scale_factor != 1:
                transform_text.append(f'×{int(scale_factor)}')
            ax.text(0.02, 0.98, f"({', '.join(transform_text)})", 
                    transform=ax.transAxes, 
                    fontsize=8, 
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))        

    # Add overall title and subtitle with adjusted positions
    fig.suptitle('Training Metrics Over Time', 
                fontsize=16, 
                weight='bold', 
                y=0.95)  # Lower the main title slightly

    subtitle = f'λ_h = {lam_h}, λ_l1 = {lam_l1}, Weights Learning Rate = {w_learning_rate}'
    fig.text(0.5, 0.90,  # Adjusted y position for the subtitle
            subtitle, 
            horizontalalignment='center',
            fontsize=12,
            style='italic')

    # Adjust layout to make more room for title and subtitle
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)


    # Save and show the figure as a .pdf file
    plt.savefig(os.path.join(save_path, 'training_metrics.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

    ########################## Plotting the DAG comparison  ##########################

    # Create a separate figure for the adjacency matrices comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Use a better colormap - options:
    # 'YlOrBr' - Yellow-Orange-Brown (good for sparse matrices)
    # 'viridis' - Perceptually uniform, colorblind-friendly
    # 'Greys' - Black and white, professional
    # 'YlGnBu' - Yellow-Green-Blue, professional
    cmap = 'YlOrBr'  # Choose one of the above

    # Plot estimated adjacency matrix (now on the left)
    im1 = ax1.imshow(compute_binary_adjacency(W), cmap=cmap, interpolation='nearest')
    ax1.set_title('Estimated DAG', pad=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlabel('Node', labelpad=10)
    ax1.set_ylabel('Node', labelpad=10)

    # Plot true adjacency matrix (now on the right)
    im2 = ax2.imshow(B_true, cmap=cmap, interpolation='nearest')
    ax2.set_title('True DAG', pad=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xlabel('Node', labelpad=10)
    ax2.set_ylabel('Node', labelpad=10)

    # Add overall title
    fig.suptitle('Estimated vs True DAG Structure', 
                fontsize=16, 
                weight='bold', 
                y=1.05)

    # Add grid lines to better separate the nodes
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(-.5, B_true.shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-.5, B_true.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Improve layout
    plt.tight_layout()

    # Save the comparison plot as a .pdf file at the specified location
    plt.savefig(os.path.join(save_path, 'dag_comparison.pdf'), 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.show()

    # %%
    # Now use a threshold of 0.3 to binarize the weighted adjacency matrix W
    W_est = np.array(model.get_W())
    B_est = compute_binary_adjacency(W_est, threshold=0.3)

    # Check if the estimated binary adjacency matrix B_est is a DAG
    is_dag_B_est = is_dag_nx(B_est)
    print(f"Is the estimated binary adjacency matrix a DAG? {is_dag_B_est}")

    print()

    # Compute the h_reg term for the true weighted adjacency matrix W_true
    h_reg_true = compute_h_reg(W_true)
    print(f"The h_reg term for the TRUE weighted adjacency matrix W_true is: {h_reg_true:.4f}")
    # Compute the h_reg term for the estimated weighted adjacency matrix W_est
    h_reg_est = compute_h_reg(W_est)
    print(f"The h_reg term for the EST weighted adjacency matrix W_est is: {h_reg_est:.4f}")

    print()

    # print the number of edges in the true graph and the estimated graph
    print(f"The number of edges in the TRUE graph: {np.sum(B_true)}")
    print(f"The number of edges in the EST graph: {np.sum(B_est)}")

    # print first 5 rows and columsn of W_est and round values to 4 decimal places and show as non-scientific notation
    np.set_printoptions(precision=4, suppress=True)

    print()

    print("The first 5 rows and columns of the estimated weighted adjacency matrix W_est\n{}".format(W_est[:5, :5]))

    # now show the adjacency matrix of the true graph and the estimated graph side by side
    plot_adjacency_matrices(true_matrix=B_true, est_matrix=B_est, save_path=os.path.join(save_path, 'adjacency_matrices.png'))


    # %%
    # plot est_dag and true_dag
    GraphDAG(B_est, B_true, save_name=os.path.join(save_path, 'est_dag_true_dag.png'))
    # calculate accuracy
    met_pcx = MetricsDAG(B_est, B_true) # expects first arg to be the predicted labels and the second arg to be the true labels
    print(met_pcx.metrics)

    # %%
    # Set the style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjusting layout to 1 row and 3 columns
    fig.suptitle('Performance Metrics Over Epochs', fontsize=16, weight='bold')

    # Plot the Cyclic F1 score
    sns.lineplot(x=range(len(F1s_cycles)), y=F1s_cycles, ax=axs[0], color=palette[0])
    axs[0].set_xlabel("Epoch", fontsize=12)
    axs[0].set_ylabel("F1 Cyclic", fontsize=12)
    axs[0].grid(True)

    # Plot the Cyclic SHD
    sns.lineplot(x=range(len(SHDs)), y=SHDs_cyclic, ax=axs[1], color=palette[2])
    axs[1].set_xlabel("Epoch", fontsize=12)
    axs[1].set_ylabel("SHD Cyclic", fontsize=12)
    axs[1].grid(True)

    # Plot the vanilla SHD
    sns.lineplot(x=range(len(SHDs)), y=SHDs, ax=axs[2], color=palette[1])
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel("SHD Structure", fontsize=12)
    axs[2].grid(True)

    # Improve layout and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
    plt.show()

    # %%
    # print first 5 rows and columsn of W_est and round values to 4 decimal places and show as non-scientific notation
    np.set_printoptions(precision=4, suppress=True)

    print("The first 4 rows and columns of the estimated weighted adjacency matrix W_est\n{}".format(W_est[:11, :11]))

    # %%
    # now show the adjacency matrix of the true graph and the estimated graph side by side
    plot_adjacency_matrices(B_true, B_est)

    # print the number of edges in the true graph and the estimated graph
    print(f"The number of edges in the estimated graph: {np.sum(B_est)}")
    print(f"The number of edges in the true graph: {np.sum(B_true)}")

    # %%
    # plot est_dag and true_dag
    GraphDAG(B_est, B_true)
    # calculate accuracy
    met_pcax = MetricsDAG(B_est, B_true)
    print(met_pcax.metrics)

    # print experiment name
    print(exp_name)

    ############################################################################################
    # benchmark model 1 nodags

    # the below setup is taken from the nodags-flows/validation /run_synthetic_benchmark.py file which runs on linear data

    X_torch = torch.tensor(X, dtype=torch.float64)

    # Initialize the NODAGS model with linear functions
    resblock = resflow_train_test_wrapper(
        n_nodes=X.shape[1],
        batch_size=128,
        l1_reg=True, # default is False
        lambda_c=5e-2, # default is 1e-2
        fun_type='lin-mlp',  # Linear version
        act_fun='none', # No activation function
        lr=1e-3, # default learning rate is 1e-3
        lip_const=0.99,
        epochs=50, # default number of epochs is 10
        optim='adam', # default optimizer is 'sgd'
        n_hidden=0, # default number of hidden layers is 1
        thresh_val=0.3, # default threshold value is 0.01
        centered=False, # default is True
        v=True, # verbose
        inline=True, # more verbose
        upd_lip=True, # default is True
        full_input=False, # default is False
    )

    # Train the model with only observational data
    resblock.train([X], [[]], batch_size=128)

    # get the weighted adjacency matrix
    W_est_nodags = np.array(resblock.get_adjacency())
    # get the binary adjacency matrix
    B_est_nodags = (W_est_nodags > 0.3).astype(int)

    # plot est_dag and true_dag
    GraphDAG(B_est_nodags, B_true)
    # calculate accuracy
    met_nodags = MetricsDAG(B_est_nodags, B_true) # expects first arg to be the predicted labels and the second arg to be the true labels
    print(met_nodags.metrics)

    ############################################################################################
    # benchmark model 2 LiNGD
    
    ling = LiNGD(k=5, random_state=seed)
    ling.fit(X)
    W_ling = np.array(ling._adjacency_matrices[0].T) # 0 to get first candidate adjacency matrix, Transpose to make each row correspond to parents
    B_est_ling = compute_binary_adjacency(W_ling)

    # plot est_dag and true_dag
    GraphDAG(B_est_ling, B_true)
    # calculate accuracy
    met_ling = MetricsDAG(B_est_ling, B_true)
    print(met_ling.metrics)

    ############################################################################################
    # benchmark model 3 DGLearn

    tabu_length = 4
    patience = 4
    #max_iter = np.inf
    max_iter = 20

    manager = CyclicManager(X, bic_coef=0.5, num_workers = 200)
    B_est_dglearn, best_score, log = tabu_search(manager, tabu_length, patience, max_iter=max_iter, first_ascent=False, verbose=1) # returns a binary matrix as learned support

    # perform virtual edge correction
    print("virtual edge correction...")
    B_est_dglearn = virtual_refine(manager, B_est_dglearn, patience=0, max_iter=max_iter, max_path_len=6, verbose=1)

    # remove any reducible edges
    B_est_dglearn = reduce_support(B_est_dglearn, fill_diagonal=False) # same as done in FRP paper
    B_est_dglearn = np.array(B_est_dglearn)

    # plot est_dag and true_dag
    GraphDAG(B_est_dglearn, B_true)
    # calculate accuracy
    met_dglearn = MetricsDAG(B_est_dglearn, B_true)
    print(met_dglearn.metrics)

    ############################################################################################
    # benchmark model 4 FRP

    edge_penalty = 0.5 * np.log(X.shape[0]) / X.shape[0] # BIC choice
    frp_result = run_filter_rank_prune(
            X, 
            loss_type="kld",
            reg_type="scad",
            reg_params={"lam": edge_penalty, "gamma": 3.7},
            edge_penalty=edge_penalty,
            n_inits=5,
            n_threads=200,
            parcorr_thrs=0.1 * (X.shape[0] / 1000)**(-1/4),
            use_loss_cache=True,
            seed=seed,
            verbose=True,
    )

    B_est_frp = np.array(frp_result["learned_support"].astype(int))

    # plot est_dag and true_dag
    GraphDAG(B_est_frp, B_true)
    # calculate accuracy
    met_frp = MetricsDAG(B_est_frp, B_true)
    print(met_frp.metrics)

    ############################################################################################
    # benchmark model 5 NOTEARS

    nt = Notears(lambda1=5e-2, h_tol=1e-5, max_iter=100, no_dag_constraint=True, loss_type='l2', seed=seed) # default loss_type is 'l2'
    #nt = Notears(lambda1=lambda1=5e-2, h_tol=1e-5, max_iter=100, no_dag_constraint=True, loss_type='laplace', seed=seed)
    #nt = Notears(lambda1=lambda1=5e-2, h_tol=1e-5, max_iter=100, no_dag_constraint=True, loss_type='logistic', seed=seed)
    nt.learn(X)

    # plot est_dag and true_dag
    W_est_nt = np.array(nt.weight_causal_matrix)
    B_est_nt = np.array(nt.causal_matrix)
    GraphDAG(B_est_nt, B_true)

    # calculate accuracy
    met_nt = MetricsDAG(B_est_nt, B_true)
    print(met_nt.metrics)

    ############################################################################################
    # benchmark model 6 GOLEM

    gol = GOLEM(num_iter=50000, lambda_1=5e-2, lambda_2=0.0, equal_variances=True, device_type='gpu', no_dag_constraint=True, seed=seed) # we deactivate DAG constraint by setting lambda_2=0.0 as done in FRP paper
    #gol = GOLEM(num_iter=50000, lambda_1=5e-2, lambda_2=0.0, device_type='gpu', no_dag_constraint=True, seed=seed) # we deactivate DAG constraint by setting lambda_2=0.0 as done in FRP paper
    gol.learn(X)

    # plot est_dag and true_dag
    W_est_gol = np.array(gol.weight_causal_matrix)
    B_est_gol = np.array(gol.causal_matrix)
    GraphDAG(B_est_gol, B_true)

    # calculate accuracy
    met_gol = MetricsDAG(B_est_gol, B_true)
    print(met_gol.metrics)

    ############################################################################################

    print("✅ ALL METHODS FITTED")

    ###### PLACE METRIC COMPUTATION CODE HERE ######

    # Compute BIC and AIC scores robustly
    bic_values = [safe_compute(compute_model_fit, B, X, default_value=None)[0] 
                  for B in [B_est, B_est_ling, B_est_dglearn, B_est_frp, B_est_nt, B_est_gol, B_est_nodags]]

    aic_values = [safe_compute(compute_model_fit, B, X, default_value=None)[1] 
                  for B in [B_est, B_est_ling, B_est_dglearn, B_est_frp, B_est_nt, B_est_gol, B_est_nodags]]

    # Compute SHD and cycle F1 robustly
    shd_cyclic = [safe_compute(compute_cycle_SHD, B_true_EC, B, default_value=None) 
                  for B in [B_est, B_est_ling, B_est_dglearn, B_est_frp, B_est_nt, B_est_gol, B_est_nodags]]

    cycle_f1 = [safe_compute(compute_cycle_F1, B_true, B, default_value=None) 
                for B in [B_est, B_est_ling, B_est_dglearn, B_est_frp, B_est_nt, B_est_gol, B_est_nodags]]

    # Compute CSS scores robustly
    css_scores = [safe_compute(compute_CSS, shd, f1, epsilon=1e-8, default_value=None) 
                  for shd, f1 in zip(shd_cyclic, cycle_f1)]

    # Compute cycle KLD robustly
    cycle_kld = [safe_compute(compute_cycle_KLD, prec_matrix, B, default_value=None)  
                for B in [B_est, B_est_ling, B_est_dglearn, B_est_frp, B_est_nt, B_est_gol, B_est_nodags]]

    ##############################################

    # Store results in a dictionary
    results = {
        "seed": seed,
        "method": ["Our", "LinG", "dglearn", "FRP", "NOTEARS", "GOLEM", "NODAGS"],
        "BIC": bic_values,
        "AIC": aic_values,
        "SHD_cyclic": shd_cyclic,
        "Cycle_F1": cycle_f1,
        "CSS": css_scores,
        "Cycle_KLD": cycle_kld
    }

    return results

# Function to save results
def save_results(results_list, output_file):
    """Merge results from different runs and save to a JSON file robustly."""
    
    print("📁 Merging and saving results...")

    try:
        # Convert results list into DataFrame
        df = pd.concat([pd.DataFrame(res) for res in results_list], ignore_index=True)

        # Convert NaN and Inf values to None before saving
        df = df.replace([np.nan, np.inf, -np.inf], None)

        # Save results to JSON
        df.to_json(output_file, orient="records", indent=4)
        print(f"✅ Merged results saved to {output_file}")

    except Exception as e:
        print(f"⚠️ Error saving results: {e}")


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
    from causal_helpers import is_dag_nx, MAE, compute_binary_adjacency, compute_h_reg, notears_dag_constraint, dagma_dag_constraint
    from causal_helpers import simulate_dag, simulate_parameter, simulate_linear_sem, simulate_linear_sem_cyclic
    from causal_helpers import load_adjacency_matrix, set_random_seed, plot_adjacency_matrices
    from causal_helpers import load_graph, load_adjacency_matrix
    from causal_metrics import compute_F1_directed, compute_F1_skeleton, compute_AUPRC, compute_AUROC, compute_model_fit
    from causal_metrics import compute_cycle_F1, compute_cycle_SHD, compute_cycle_KLD, compute_CSS  # CSS: Cyclic Structure Score
    from connectome_cyclic_data_generator import sample_cyclic_data

    #################### NODAGS #########################

    from nodags_flows.models.resblock_trainer import resflow_train_test_wrapper

    ##################### DGLearn ############################

    from dglearn.dglearn.dg.adjacency_structure import AdjacencyStucture
    from dglearn.dglearn.dg.graph_equivalence_search import GraphEquivalenceSearch
    from dglearn.dglearn.dg.converter import binary2array
    from dglearn.dglearn.learning.cyclic_manager import CyclicManager
    from dglearn.dglearn.dg.reduction import reduce_support
    from dglearn.dglearn.learning.search.virtual import virtual_refine
    from dglearn.dglearn.learning.search.tabu import tabu_search

    ###################### LiNGD #########################

    from lingd import LiNGD

    ##################### FRP ###########################

    from frp.run_causal_discovery import  run_filter_rank_prune, run_dglearn

    #####################################################

    # causal libraries
    import cdt, castle
    from castle.algorithms import GOLEM, Notears

    # causal metrics
    from cdt.metrics import precision_recall, SHD, SID
    from castle.metrics import MetricsDAG
    from castle.common import GraphDAG
    from causallearn.graph.SHD import SHD as SHD_causallearn

    ##################################################################
    ############################ START ###############################
    ##################################################################

    # Define dataset paths based on experiment type and graph type
    data_dir = "/share/amine.mcharrak/cyclic_data_final"
    num_seeds = args.num_seeds
    seed_values = list(range(1, num_seeds + 1))

    # Define base paths for different experiments and graph types
    experiment_paths = {
        "increase_k": {
            "ER": [
                "10ER10_linear_cyclic_GAUSS-EV_seed_1",
                "10ER20_linear_cyclic_GAUSS-EV_seed_1",
                "10ER30_linear_cyclic_GAUSS-EV_seed_1",
                "10ER41_linear_cyclic_GAUSS-EV_seed_1",
            ],
            "SF": [
                "10SF10_linear_cyclic_GAUSS-EV_seed_1",
                "10SF20_linear_cyclic_GAUSS-EV_seed_1",
                "10SF30_linear_cyclic_GAUSS-EV_seed_1",
                "10SF41_linear_cyclic_GAUSS-EV_seed_1",
            ],
            "NWS": [
                "10NWS11_linear_cyclic_GAUSS-EV_seed_1",
                "10NWS20_linear_cyclic_GAUSS-EV_seed_1",
                "10NWS31_linear_cyclic_GAUSS-EV_seed_1",
                "10NWS41_linear_cyclic_GAUSS-EV_seed_1",
            ],
        },
        "increase_d": {
            "ER": [
                "10ER30_linear_cyclic_GAUSS-EV_seed_1",
                "15ER46_linear_cyclic_GAUSS-EV_seed_1",
                "20ER59_linear_cyclic_GAUSS-EV_seed_1",
            ],
            "SF": [
                "10SF30_linear_cyclic_GAUSS-EV_seed_1",
                "15SF45_linear_cyclic_GAUSS-EV_seed_1",
                "20SF60_linear_cyclic_GAUSS-EV_seed_1",
            ],
            "NWS": [
                "10NWS31_linear_cyclic_GAUSS-EV_seed_1",
                "15NWS44_linear_cyclic_GAUSS-EV_seed_1",
                "20NWS60_linear_cyclic_GAUSS-EV_seed_1",
            ],
        },
        "increase_n": {
            "ER": ["10ER30_linear_cyclic_GAUSS-EV_seed_1"],
            "SF": ["10SF30_linear_cyclic_GAUSS-EV_seed_1"],
            "NWS": ["10NWS31_linear_cyclic_GAUSS-EV_seed_1"],
        },
    }

    # Check if the specified experiment and graph type exist
    if args.exp_name not in experiment_paths or args.graph_type not in experiment_paths[args.exp_name]:
        print(f"⚠️ Invalid combination of experiment name '{args.exp_name}' and graph type '{args.graph_type}'. Exiting.")
        exit(1)

    base_paths = [os.path.join(data_dir, path) for path in experiment_paths[args.exp_name][args.graph_type]]

    # Define sample sizes for each experiment type
    num_samples_dict = {
        "increase_k": [2000],  # Fixed sample size for k experiments
        "increase_d": [2000],  # Fixed sample size for d experiments
        "increase_n": [500, 1000, 2000, 5000],  # Varying sample sizes for n experiments
    }
    num_samples_list = num_samples_dict[args.exp_name]

    # Run experiments and store results
    output_dir = os.path.join(data_dir, "experiment_results", args.exp_name, args.graph_type)
    os.makedirs(output_dir, exist_ok=True)

    for base_path in base_paths:
        base_folder = os.path.basename(base_path)
        path_output_dir = os.path.join(output_dir, base_folder)
        os.makedirs(path_output_dir, exist_ok=True)

        # Loop over sample sizes first so that we save one JSON file per sample size.
        for num_samples in num_samples_list:
            results_all = []  # create a new list for this sample size
            for seed in seed_values:
                print(f"🚀 Running experiment: {args.exp_name}, Graph: {args.graph_type}, Base: {base_folder}, Seed: {seed}, Samples: {num_samples}")
                results = run_experiment(seed, base_path, num_samples)
                # Add sample size info to the result dictionary (if run_experiment doesn't already do it)
                results["num_samples"] = num_samples
                results_all.append(results)
                
            # Save results for this sample size in a separate JSON file.
            output_file = os.path.join(path_output_dir, f"experiment_results_n{num_samples}.json")
            save_results(results_all, output_file)

    print(f"✅ All experiments completed for {args.exp_name} with graph type {args.graph_type}!")
