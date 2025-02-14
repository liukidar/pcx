import multiprocessing as mp
mp.set_start_method("fork", force=True)
#mp.set_start_method("spawn", force=True)
#mp.set_start_method("forkserver", force=True)

import os
import argparse
import pandas as pd
import json
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import timeit

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
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["JAX_PLATFORM_NAME"] = "cuda"  # Force JAX to use CUDA

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

# --- Define run_experiment ---
def run_experiment(seed, sample_size, data_path, base_folder=None):
    """
    Runs the experiment for a given seed and sample size.
    For exp_name "mixed_confounding": all files are in data_path.
    For "increase_n_disc": base_folder is provided (global files are in base_folder)
    and data_path points to one of the n_disc_* subfolders containing training files.
    Returns a dictionary with computed metrics.
    """
    
    set_random_seed(seed)
    print(f"🔄 Running experiment with seed {seed}")

    global var_dict
    var_dict = {}  # Reset any global variable if needed

    if base_folder is not None:
        # For "increase_n_disc": load global matrices from base_folder
        adj_matrix = pd.read_csv(os.path.join(base_folder, 'adj_matrix.csv'), header=None)
        weighted_adj_matrix = pd.read_csv(os.path.join(base_folder, 'W_adj_matrix.csv'), header=None)
        # And load the training data file from the subfolder
        train_file = os.path.join(data_path, f"train_n_samples_{sample_size}.csv")
    else:
        # For "mixed_confounding": all files are in data_path
        adj_matrix = pd.read_csv(os.path.join(data_path, 'adj_matrix.csv'), header=None)
        weighted_adj_matrix = pd.read_csv(os.path.join(data_path, 'W_adj_matrix.csv'), header=None)
        train_file = os.path.join(data_path, "train.csv")
    
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

    # Get weighted adjacency matrix
    W = model.get_W()
    print("This is the weighted adjacency matrix:\n", W)
    print()
    print("The shape of the weighted adjacency matrix is: ", W.shape)
    print()
    #print(model)
    print()

    # Check if all nodes are frozen initially
    print("Initially, are all nodes frozen?:", model.are_vodes_frozen())
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

    # %%
    model.is_cont_node

    # %%
    # TODO: make the below params global or input to the functions in which it is used.

    w_learning_rate = 1e-3 # Notes: 5e-1 is too high
    h_learning_rate = 1e-4
    T = 1

    nm_epochs = 1500 # not much happens after 2000 epochs
    every_n_epochs = 1
    batch_size = 128


    # {'fdr': 0.0667, 'tpr': 0.9333, 'fpr': 0.0196, 'shd': 1, 'nnz': 15, 'precision': 0.9333, 'recall': 0.9333, 'F1': 0.9333, 'gscore': 0.8667}
    # bs_128_lrw_0.001_lrh_0.0001_lamh_10000.0_laml1_1.0_epochs_2000 gives F1 score of 0.9333
    # NOTE THIS USES EPSILON=1e-8 FOR THE NORMALIZATION OF THE REGULARIZATION TERMS
    # lam_h = 1e4
    # lam_l1 = 1e0

    lam_h = 1e4
    lam_l1 = 1e0

    # Create a file name string for the hyperparameters
    exp_name = f"bs_{batch_size}_lrw_{w_learning_rate}_lrh_{h_learning_rate}_lamh_{lam_h}_laml1_{lam_l1}_epochs_{nm_epochs}"
    print("Name of the experiment: ", exp_name)

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

    # %%
    # for reference compute the MAE, SID, and SHD between the true adjacency matrix and an all-zero matrix and then print it
    # this acts as a baseline for the MAE, SID, and SHD similar to how 1/K accuracy acts as a baseline for classification tasks where K is the number of classes

    W_zero = np.zeros_like(W_true)
    print("MAE between the true adjacency matrix and an all-zero matrix: ", MAE(W_true, W_zero))
    print("SHD between the true adjacency matrix and an all-zero matrix: ", SHD(B_true, compute_binary_adjacency(W_zero)))
    #print("SID between the true adjacency matrix and an all-zero matrix: ", SID(W_true, W_zero))

    # %%
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
        #optim_w = pxu.Optim(lambda: optax.adam(w_learning_rate), pxu.M(pxnn.LayerParam)(model))
        optim_w = pxu.Optim(lambda: optax.adamw(w_learning_rate, nesterov=True), pxu.M(pxnn.LayerParam)(model))

    # %%
    # Initialize lists to store differences and energies
    MAEs = []
    SHDs = []
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

    SHD_init = SHD(B_true, B_init)
    print(f"Start SHD between B_true and B_init: {SHD_init:.4f}")

    F1_init = compute_F1_directed(B_true, B_init)
    print(f"Start F1 between B_true and B_init: {F1_init:.4f}")

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

            # Compute metrics every n epochs
            if (epoch + 1) % every_n_epochs == 0 or epoch == 0:
                MAEs.append(float(MAE(W_true, W)))
                SHDs.append(float(SHD(B_true, compute_binary_adjacency(W), double_for_anticausal=False)))
                F1s.append(float(compute_F1_directed(B_true, B)))
                pc_energies.append(float(epoch_pc_energy))
                l1_regs.append(float(epoch_l1_reg))
                epoch_h_reg_raw = compute_h_reg(W)
                h_regs.append(float(epoch_h_reg_raw))
                objs.append(float(epoch_obj))

                # Update progress bar with the current status
                pbar.set_description(f"MAE: {MAEs[-1]:.4f}, F1: {F1s[-1]:.4f}, SHD: {SHDs[-1]:.4f} || PC Energy: {pc_energies[-1]:.4f}, L1 Reg: {l1_regs[-1]:.4f}, H Reg: {h_regs[-1]:.4f}, Obj: {objs[-1]:.4f}")


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
    # Define experiment name
    exp_name = f"bs_{batch_size}_lrw_{w_learning_rate}_lrh_{h_learning_rate}_lamh_{lam_h}_laml1_{lam_l1}_epochs_{nm_epochs}"

    # Create subdirectory in linear folder with the name stored in exp_name
    save_path = os.path.join('plots/linear_mixed_disc_x0', exp_name)
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
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    # Create a figure and subplots using GridSpec
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # Adjust layout to make more room for title and subtitle
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)

    # Create axes
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2:])
    # ax12 spans two columns

    axes = [ax00, ax01, ax02, ax03, ax10, ax11, ax12]

    # Plot configurations
    plot_configs = [
        {'metric': MAEs, 'title': 'Mean Absolute Error', 'ylabel': 'MAE', 'color': '#2ecc71', 'ax': ax00},
        {'metric': SHDs, 'title': 'Structural Hamming Distance', 'ylabel': 'SHD', 'color': '#e74c3c', 'ax': ax01},
        {'metric': F1s, 'title': 'F1 Score', 'ylabel': 'F1', 'color': '#3498db', 'ax': ax02},
        {'metric': pc_energies, 'title': 'PC Energy', 'ylabel': 'Energy', 'color': '#9b59b6', 'ax': ax03},
        {'metric': l1_regs, 'title': 'L1 Regularization', 'ylabel': 'L1', 'color': '#f1c40f', 'ax': ax10},
        {'metric': h_regs, 'title': 'DAG Constraint', 'ylabel': 'h(W)', 'color': '#e67e22', 'ax': ax11},
        {'metric': objs, 'title': 'Total Objective', 'ylabel': 'Loss', 'color': '#1abc9c', 'ax': ax12}
    ]

    # Create all subplots
    for config in plot_configs:
        ax = config['ax']
        
        # Plot data with rolling average
        epochs = range(len(config['metric']))
        
        # Determine if we should use log scale and/or scaling factor
        use_log_scale = config['title'] in ['PC Energy', 'Total Objective']
        scale_factor = 1e4 if config['title'] == 'DAG Constraint' else 1
        
        # Apply scaling and/or log transform to the metric
        metric_values = np.array(config['metric'])
        if use_log_scale:
            # Add small constant to avoid log(0)
            metric_values = np.log10(np.abs(metric_values) + 1e-10)
        metric_values = metric_values * scale_factor
        
        # Plot raw data
        raw_line = ax.plot(epochs, metric_values, 
                        alpha=0.3, 
                        color=config['color'], 
                        label='Raw')
        
        # Calculate and plot rolling average
        window_size = 50
        if len(metric_values) > window_size:
            rolling_mean = np.convolve(metric_values, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size-1, len(metric_values)), 
                    rolling_mean, 
                    color=config['color'], 
                    linewidth=2, 
                    label='Moving Average')
        
        # Customize each subplot
        ax.set_title(config['title'], pad=10)
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
        
        # Add legend if it's the total objective plot
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
                y=0.98)

    subtitle = f'λ_h = {lam_h}, λ_l1 = {lam_l1}, Weights Learning Rate = {w_learning_rate}'
    fig.text(0.5, 0.93, 
            subtitle, 
            horizontalalignment='center',
            fontsize=12,
            style='italic')

    # Save and show the figure as a .pdf file at the specified location
    plt.savefig(os.path.join(save_path, 'training_metrics.pdf'), 
                bbox_inches='tight', 
                dpi=300)
    plt.show()

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

    # %%
    # Check if B_est is indeed a DAG
    def is_dag(adjacency_matrix):
        """
        Check if a given adjacency matrix represents a Directed Acyclic Graph (DAG).
        
        Parameters:
            adjacency_matrix (numpy.ndarray): A square matrix representing the adjacency of a directed graph.
            
        Returns:
            bool: True if the graph is a DAG, False otherwise.
        """
        # Create a directed graph from the adjacency matrix
        graph = nx.DiGraph(adjacency_matrix)
        
        # Check if the graph is a DAG
        return nx.is_directed_acyclic_graph(graph)

    # Check if the estimated binary adjacency matrix B_est is a DAG
    is_dag_B_est = is_dag(B_est)
    print(f"Is the estimated binary adjacency matrix a DAG? {is_dag_B_est}")


    # Compute the h_reg term for the true weighted adjacency matrix W_true
    h_reg_true = compute_h_reg(W_true)
    print(f"The h_reg term for the true weighted adjacency matrix W_true is: {h_reg_true:.4f}")

    # Compute the h_reg term for the estimated weighted adjacency matrix W_est
    h_reg_est = compute_h_reg(W_est)
    print(f"The h_reg term for the estimated weighted adjacency matrix W_est is: {h_reg_est:.4f}")

    # print first 5 rows and columsn of W_est and round values to 4 decimal places and show as non-scientific notation
    np.set_printoptions(precision=4, suppress=True)

    print("The first 5 rows and columns of the estimated weighted adjacency matrix W_est\n{}".format(W_est[:5, :5]))

    # now show the adjacency matrix of the true graph and the estimated graph side by side
    plot_adjacency_matrices(true_matrix=B_true, est_matrix=B_est, save_path=os.path.join(save_path, 'adjacency_matrices.png'))

    # print the number of edges in the true graph and the estimated graph
    print(f"The number of edges in the true graph: {np.sum(B_true)}")
    print(f"The number of edges in the estimated graph: {np.sum(B_est)}")

    # %%
    # plot est_dag and true_dag
    GraphDAG(B_est, B_true, save_name=os.path.join(save_path, 'est_dag_true_dag.png'))
    # calculate accuracy
    met_pcx = MetricsDAG(B_est, B_true)
    print(met_pcx.metrics)

    # %%
    # Set the style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Adjusting layout to 1 row and 3 columns
    fig.suptitle('Performance Metrics Over Epochs', fontsize=16, weight='bold')

    # Plot the MAE
    sns.lineplot(x=range(len(MAEs)), y=MAEs, ax=axs[0], color=palette[0])
    axs[0].set_title("Mean Absolute Error (MAE)", fontsize=14)
    axs[0].set_xlabel("Epoch", fontsize=12)
    axs[0].set_ylabel("MAE", fontsize=12)
    axs[0].grid(True)

    # Plot the SHD
    sns.lineplot(x=range(len(SHDs)), y=SHDs, ax=axs[1], color=palette[2])
    axs[1].set_title("Structural Hamming Distance (SHD)", fontsize=14)
    axs[1].set_xlabel("Epoch", fontsize=12)
    axs[1].set_ylabel("SHD", fontsize=12)
    axs[1].grid(True)

    # Plot the Energy
    sns.lineplot(x=range(len(pc_energies)), y=pc_energies, ax=axs[2], color=palette[3])
    axs[2].set_title("Energy", fontsize=14)
    axs[2].set_xlabel("Epoch", fontsize=12)
    axs[2].set_ylabel("Energy", fontsize=12)
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
    print(f"The number of edges in the true graph: {np.sum(B_true)}")
    print(f"The number of edges in the estimated graph: {np.sum(B_est)}")

    # plot est_dag and true_dag
    GraphDAG(B_est, B_true)
    # calculate accuracy
    met_pcax = MetricsDAG(B_est, B_true)
    print(met_pcax.metrics)

    # print experiment name
    print(exp_name)

    ############### benchmark models ################

    ##################################################################################
    # benchmark model 1 notears

    # notears learn
    nt = Notears(lambda1=0.1, h_tol=1e-5, max_iter=200, loss_type='l2', seed=seed) # takes 8 seconds, final loss @ 4.1
    #nt = Notears(lambda1=0.1, h_tol=1e-5, max_iter=200, loss_type='laplace', seed=seed) # takes 16 seconds, final loss @ 8.2
    #nt = Notears(lambda1=0.1, h_tol=1e-3, max_iter=100, loss_type='logistic', seed=seed) # takes over 5 minutes, final loss @ 245
    #nt = Notears(lambda1=0.1, h_tol=1se-3, max_iter=100, loss_type='poisson', seed=seed)
    nt.learn(X)

    # plot est_dag and true_dag
    W_est_nt = np.array(nt.weight_causal_matrix)
    B_est_nt = np.array(nt.causal_matrix)
    GraphDAG(B_est_nt, B_true)

    # calculate accuracy
    met_nt = MetricsDAG(B_est_nt, B_true)
    print(met_nt.metrics)

    ##################################################################################
    # benchmark model 2 pc (Peter & Clark)
    
    # A variant of PC-algorithm, one of [`original`, `stable`, `parallel`]
    pc = PC(variant='parallel', alpha=0.03, ci_test='fisherz', random_state=seed) # F1 of 75%
    #pc = PC(variant='stable', alpha=0.03) # F1 of 66%
    #pc = PC(variant='original', alpha=0.03) # F1 of 55%
    pc.learn(X)

    # plot est_dag and true_dag
    B_est_pc = np.array(pc.causal_matrix)
    GraphDAG(B_est_pc, B_true)

    # calculate accuracy
    met_pc = MetricsDAG(B_est_pc, B_true)
    print(met_pc.metrics)

    ##################################################################################
    # benchmark model 3 icalingam

    # max_iter : int, optional (default=1000)
    g_ICAlingam = ICALiNGAM(max_iter=2000, random_state=seed) # F1 of 71%
    g_ICAlingam.learn(X)

    # plot est_dag and true_dag
    B_est_icalingam = np.array(g_ICAlingam.causal_matrix)
    W_est_icalingam = np.array(g_ICAlingam.weight_causal_matrix)
    GraphDAG(B_est_icalingam, B_true)

    # calculate accuracy
    met_icalingam = MetricsDAG(B_est_icalingam, B_true)
    print(met_icalingam.metrics)

    ##################################################################################
    # benchmark model 4 lim

    # LiM from lingam library
    lim = lingam.LiM(w_threshold=0.3, max_iter=200, h_tol=1e-8, rho_max=1e12, lambda1=0.4, seed=seed) # default rho_max is 1e16
    # create array of shape (1, n_features) which indicates which columns/variables in data.values are discrete or continuous variables, where "1" indicates a continuous variable, while "0" a discrete variable.
    is_cont = np.array([(1 if X_df[col].nunique() > 2 else 0) for col in X_df.columns]).reshape(1, -1)
    #LiM.fit(X=X, dis_con=is_cont, only_global=False) # does not finish, even fater 375 minutes
    lim.fit(X=X, dis_con=is_cont, only_global=True)

    # plot est_dag and true_dag
    W_est_lim = np.array(lim.adjacency_matrix_)
    B_est_lim = compute_binary_adjacency(W_est_lim)

    GraphDAG(B_est_lim, B_true)

    # calculate accuracy
    met_lim = MetricsDAG(B_est_lim, B_true)
    print(met_lim.metrics)

    ##################################################################################
    # benchmark model 5 mCMIkNN

    # Import PCParallel from your module.
    from pc_parallel import PCParallel

    # Set parameters for the PCParallel-based test.
    alpha = 0.05       # Significance level, increasing it will increase the number of edges
    processes = 100    # Number of processes
    kcmi = 25          # k for mutual information estimation
    kperm = 5          # Local permutation neighborhood size
    Mperm = 100        # Number of permutations for p-value estimation
    subsample = 500    # Subsample parameter (if desired) # 1 min
    #subsample = 1000   # 2 mins
    max_level = None   # Maximum conditioning set size (None == infinite)

    # Create an instance of PCParallel; it automatically builds an mCMIkNN estimator.
    pc_parallel_instance = PCParallel(
        alpha=alpha,
        processes=processes,
        kcmi=kcmi,
        kperm=kperm,
        Mperm=Mperm,
        max_level=max_level
    )
    # Set the subsample parameter on the estimator if needed.
    pc_parallel_instance.estimator.subsample = subsample

    # Run the parallel PC algorithm on the loaded DataFrame.
    graph, sepsets = pc_parallel_instance.run(X_df)

    # Convert the returned NetworkX graph to a NumPy array.
    B_est_mcmiknn = nx.to_numpy_array(graph)

    GraphDAG(B_est_mcmiknn, B_true)

    # calculate accuracy
    met_mcmiknn = MetricsDAG(B_est_mcmiknn, B_true)

    print(met_mcmiknn.metrics)

    ###### METRIC COMPUTATION ######

    # Define models and their corresponding adjacency/weight matrices
    models = {
        "Our (PC)": (B_est, W_est),
        "Peter Clark": (B_est_pc, None),  
        "NOTEARS": (B_est_nt, W_est_nt),
        "ICALiNGAM": (B_est_icalingam, W_est_icalingam),
        "LiM": (B_est_lim, W_est_lim),
        "mCMIkNN": (B_est_mcmiknn, None),  
    }

    # Indices for treatment (intervention) and target variables
    if args.exp_name == "mixed_confounding":
        interv_node_idx = 0
        target_node_idx = 11
    elif args.exp_name == "increase_n_disc":
        interv_node_idx, target_node_idx = pick_source_target(B_true, random_state=seed)
    else:
        raise ValueError(f"Invalid experiment name: {args.exp_name}")
    
    # Initialize results dictionary with metadata
    results = {
        "seed": seed,
        "sample_size": sample_size,
        "methods": list(models.keys()),
    }

    # Store metrics as key-value pairs where each metric maps method names to their values
    results["SHD_absolute"] = {}
    results["SHD_normalized"] = {}
    results["SID_absolute"] = {}
    results["SID_normalized"] = {}
    results["AID_absolute"] = {}
    results["AID_normalized"] = {}
    results["F1_Score"] = {}
    results["TEE"] = {}
    results["AUROC"] = {}
    results["AUPRC"] = {}
    results["True_Total_Effect"] = {}
    results["Estimated_Total_Effect"] = {}

    for model_name, (B, W) in models.items():
        B_binary = (B > 0).astype(np.int8)

        # Ensure acyclicity for all models
        if not is_dag_nx(B_binary):
            print(f"🚨 Warning: {model_name} produced a cyclic B_est!")
            B_binary = enforce_dag(B_binary)  # Fix cycles minimally
            print(f"✅ {model_name} is now acyclic after enforcing DAG constraints.")

            # 🔥 Update the model dictionary with the corrected B_binary
            models[model_name] = (B_binary, W)            

        # Compute metrics
        results["SHD_normalized"][model_name], results["SHD_absolute"][model_name] = compute_SHD((B_true > 0).astype(np.int8), B_binary)
        results["SID_normalized"][model_name], results["SID_absolute"][model_name] = compute_SID((B_true > 0).astype(np.int8), B_binary)
        results["AID_normalized"][model_name], results["AID_absolute"][model_name] = compute_ancestor_AID((B_true > 0).astype(np.int8), B_binary)
        results["F1_Score"][model_name] = compute_F1_directed(B_true, B)

        # Compute Total Effect Estimation Error (TEE), AUROC, and AUPRC (if weights exist)
        if W is not None:
            true_effect, est_effect, tee = compute_TEE(W_true, W, interv_node_idx, target_node_idx)
            _B_true = (np.abs(W_true) > 0).astype(int)

            results["TEE"][model_name] = tee
            results["AUROC"][model_name] = compute_AUROC(_B_true, W)[-1]
            results["AUPRC"][model_name] = compute_AUPRC(_B_true, W)[-1]
            results["True_Total_Effect"][model_name] = true_effect
            results["Estimated_Total_Effect"][model_name] = est_effect
        else:
            results["TEE"][model_name] = None
            results["AUROC"][model_name] = None
            results["AUPRC"][model_name] = None
            results["True_Total_Effect"][model_name] = None
            results["Estimated_Total_Effect"][model_name] = None

    return results

# Function to save results
def save_results(results_list, output_file):
    """Merge results from different runs and save to a JSON file robustly."""
    
    print("📁 Merging and saving results...")

    try:
        # Convert list of result dictionaries into a pandas DataFrame
        df = pd.DataFrame(results_list)

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
    from causal_helpers import enforce_dag, pick_source_target
    from causal_helpers import is_dag_nx, MAE, compute_binary_adjacency, compute_h_reg, notears_dag_constraint, dagma_dag_constraint
    from causal_helpers import load_adjacency_matrix, set_random_seed, plot_adjacency_matrices
    from causal_helpers import load_graph, load_adjacency_matrix
    from causal_metrics import compute_F1_directed, compute_F1_skeleton, compute_AUPRC, compute_AUROC, compute_cycle_F1
    from causal_metrics import compute_SHD, compute_SID, compute_ancestor_AID, compute_TEE

    # causal libraries
    import lingam
    os.environ["CASTLE_BACKEND"] = "pytorch"  # Prevents repeated backend initialization
    import cdt, castle

    from castle.algorithms import Notears, ICALiNGAM, PC
    from castle.algorithms import DirectLiNGAM, GOLEM
    from castle.algorithms.ges.ges import GES
    from castle.algorithms import PC

    # causal metrics
    from cdt.metrics import precision_recall, SHD, SID
    from castle.metrics import MetricsDAG
    from castle.common import GraphDAG
    from causallearn.graph.SHD import SHD as SHD_causallearn

    ##################################################################
    ############################ START ###############################
    ##################################################################

    if args.exp_name == "mixed_confounding":
        # For mixed_confounding, use a fixed list of sample sizes and a fixed data path.
        sample_sizes = [500, 1000, 2000, 4000]
        seed_values = list(range(1, args.num_seeds + 1))
        data_path = "../data/custom_mixed_confounding_softplus/"
        
        output_dir = os.path.join(data_path, "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        
        for sample_size in sample_sizes:
            results_all = [run_experiment(seed, sample_size, data_path) for seed in seed_values]
            output_file = os.path.join(output_dir, f"results_n{sample_size}.json")
            save_results(results_all, output_file)
            
            del results_all
            gc.collect()
            jax.clear_caches()
            if "torch" in globals():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    elif args.exp_name == "increase_n_disc":
        # For increase_n_disc, use the provided --path_name (default is now the base folder)
        base_path = args.path_name  # e.g. "/share/amine.mcharrak/mixed_data_final/10ER30_linear_mixed_seed_1"
        base_folder = os.path.basename(base_path)
        print(f"Using base folder: {base_folder}")
        
        # Global files (adj_matrix.csv, W_adj_matrix.csv) are in base_path.
        # Training files are in the subdirectories named "n_disc_*" and depend on the n_samples value.
        sample_size = args.n_samples
        print(f"Sample size to use (n_samples): {sample_size}")
        
        # List all subdirectories starting with "n_disc_"
        disc_subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path)
                        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("n_disc_")]
        disc_subdirs.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
        print(f"Found {len(disc_subdirs)} disc subdirectories.")
        
        output_dir = os.path.join(base_path, "experiment_results")
        os.makedirs(output_dir, exist_ok=True)
        seed_values = list(range(1, args.num_seeds + 1))
        
        for disc_subdir in disc_subdirs:
            disc_label = os.path.basename(disc_subdir)  # e.g., "n_disc_5"
            results_all = []
            for seed in seed_values:
                print(f"🚀 Running experiment for {disc_label} with seed {seed}")
                # For increase_n_disc, pass base_folder as well so that global files are read from there.
                results = run_experiment(seed, sample_size, disc_subdir, base_folder=base_path)
                # Augment results with additional information
                results["seed"] = seed
                results["sample_size"] = sample_size
                results["n_disc"] = disc_label
                results_all.append(results)
            output_file = os.path.join(output_dir, f"experiment_results_{disc_label}.json")
            save_results(results_all, output_file)
            
            del results_all
            gc.collect()
            jax.clear_caches()
            if "torch" in globals():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"✅ All experiments completed for {args.exp_name}!")

    # Clear GPU memory from torch and JAX
    torch.cuda.empty_cache()
    jax.clear_backends()
    jax.clear_caches()
    print("✅ GPU memory cleared.")
