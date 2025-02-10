import os
import argparse
import pandas as pd
import multiprocessing
from multiprocessing import Pool, RawArray
import json

import csv
import logging
import multiprocessing
from multiprocessing import Pool
import networkx as nx
import numpy as np

from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations, product, chain
from math import floor

from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.stats import rankdata


parser = argparse.ArgumentParser(description="Run ICML Mixed Data Experiment")
parser.add_argument("--seed", type=int, default=2, help="Random seed for experiment")
args = parser.parse_args()

seed = args.seed

# Define output directory
output_dir = "experiment_results_mixed_data_new"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "results_all_seeds.json")

# choose the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# disable preallocation of memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# pcx
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

# 3rd party
import jax
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as jax_numpy_linalg # for expm()
import jax.scipy.linalg as jax_scipy_linalg # for slogdet()
import jax.random as random
import optax
import numpy as np
from pandas.api.types import is_float_dtype
from ucimlrepo import fetch_ucirepo 

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import timeit
from sklearn.metrics import f1_score

# own
import causal_helpers
from causal_model import Complete_Graph
from causal_helpers import is_dag_nx, MAE, compute_binary_adjacency, compute_h_reg, notears_dag_constraint, dagma_dag_constraint
from causal_helpers import load_adjacency_matrix, set_random_seed, plot_adjacency_matrices
from causal_helpers import load_graph, load_adjacency_matrix
from causal_metrics import compute_F1_directed, compute_F1_skeleton, compute_AUPRC, compute_AUROC, compute_cycle_F1
from causal_metrics import compute_SHD, compute_SID, compute_ancestor_AID, compute_TEE

# causal libraries
import lingam
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

######################### mcmiknn #########################

from causal_model import mCMIkNN
from itertools import combinations, product, chain
from multiprocessing import Pool, RawArray
import logging
from datetime import datetime

var_dict = {} # might not be needed but keep it for now

def _init_worker(data, data_shape, graph, vertices, test, alpha):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['data'] = data
    var_dict['data_shape'] = data_shape

    var_dict['graph'] = graph
    var_dict['vertices'] = vertices

    var_dict['alpha'] = alpha
    var_dict['test'] = test

def _test_worker(i, j, lvl):
    test = var_dict['test']
    alpha = var_dict['alpha']
    data_arr = np.frombuffer(var_dict['data']).reshape(var_dict['data_shape'])
    graph = np.frombuffer(var_dict['graph'], dtype="int32").reshape((var_dict['vertices'],
                                                                    var_dict['vertices']))
    
    # unconditional
    if lvl < 1:
        p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=None)
        if (p_val > alpha):
            return (i, j, p_val, [])
    # conditional
    else:
        candidates_1 = np.arange(var_dict['vertices'])[(graph[i] == 1)]
        candidates_1 = np.delete(candidates_1, np.argwhere((candidates_1==i) | (candidates_1==j)))

        if (len(candidates_1) < lvl):
            return None
        
        for S in [list(c) for c in combinations(candidates_1, lvl)]:
            p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=data_arr[:, list(S)])
            if (p_val > alpha):
                return (i, j, p_val, list(S))
            
    return None

def _unid(g, i, j):
    return g.has_edge(i, j) and not g.has_edge(j, i)

def _bid(g, i, j):
    return g.has_edge(i, j) and g.has_edge(j, i)

def _adj(g, i, j):
    return g.has_edge(i, j) or g.has_edge(j, i)

def rule1(g, j, k):
    for i in g.predecessors(j):
        # i -> j s.t. i not adjacent to k
        if _unid(g, i, j) and not _adj(g, i, k):
            g.remove_edge(k, j)
            return True
    return False

def rule2(g, i, j):
    for k in g.successors(i):
        # i -> k -> j
        if _unid(g, k, j) and _unid(g, i, k):
            g.remove_edge(j, i)
            return True
    return False

def rule3(g, i, j):
    for k, l in combinations(g.predecessors(j), 2):
        # i <-> k -> j and i <-> l -> j s.t. k not adjacent to l
        if (not _adj(g, k, l) and _bid(g, i, k) and _bid(g, i, l) and _unid(g, l, j) and _unid(g, k, j)):
            g.remove_edge(j, i)
            return True
    return False

def rule4(g, i, j):
    for l in g.predecessors(j):
        for k in g.predecessors(l):
            # i <-> k -> l -> j s.t. k not adjacent to j and i adjacent to l
            if (not _adj(g, k, j) and _adj(g, i, l) and _unid(g, k, l) and _unid(g, l, j) and _bid(g, i, k)):
                g.remove_edge(j, i)
                return True
    return False

def _direct_edges(graph, sepsets):
    digraph = nx.DiGraph(graph)
    for i in graph.nodes():
        for j in nx.non_neighbors(graph, i):
            for k in nx.common_neighbors(graph, i, j):
                sepset = sepsets[(i, j)] if (i, j) in sepsets else []
                if k not in sepset:
                    if (k, i) in digraph.edges() and (i, k) in digraph.edges():
                        digraph.remove_edge(k, i)
                    if (k, j) in digraph.edges() and (j, k) in digraph.edges():
                        digraph.remove_edge(k, j)

    bidirectional_edges = [(i, j) for i, j in digraph.edges if digraph.has_edge(j, i)]
    for i, j in bidirectional_edges:
        if _bid(digraph, i, j):
            continue
        if (rule1(digraph, i, j) or rule2(digraph, i, j) or rule3(digraph, i, j) or rule4(digraph, i, j)):
            continue

    return digraph

def parallel_stable_pc(data, estimator, alpha=0.05, processes=1, max_level=None):
    """
    Perform the Parallel Stable PC algorithm for causal discovery.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    estimator: The conditional independence (CI) test to be used.
    alpha (float): The significance level for the CI tests. Default is 0.05.
    processes (int): The number of processes to use for parallel computation. Default is 1.
    max_level (int or None): The maximum level of the PC algorithm. If None, no limit is set. Default is None.

    Returns:
    nx.DiGraph: The directed graph discovered by the algorithm.
    dict: The separation sets for the discovered edges.
    """
    cols = data.columns
    cols_map = np.arange(len(cols))

    data_raw = RawArray('d', data.shape[0] * data.shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    data_arr = np.frombuffer(data_raw).reshape(data.shape)
    # Copy data to our shared array.
    np.copyto(data_arr, data.values)

    # same magic as for data
    vertices = len(cols)
    graph_raw = RawArray('i', np.ones(vertices*vertices).astype(int))
    graph = np.frombuffer(graph_raw, dtype="int32").reshape((vertices, vertices))
    sepsets = {}

    lvls = range((len(cols) - 1) if max_level is None else min(len(cols)-1, max_level+1))
    for lvl in lvls:
        configs = [(i, j, lvl) for i, j in product(cols_map, cols_map) if i != j and graph[i][j] == 1]

        logging.info(f'Starting level {lvl} pool with {len(configs)} remaining edges at {datetime.now()}')
        with Pool(processes=processes, initializer=_init_worker,
                initargs=(data_raw, data.shape, graph_raw, vertices, estimator, alpha)) as pool:
            result = pool.starmap(_test_worker, configs)

        for r in result:
            if r is not None:
                graph[r[0]][r[1]] = 0
                graph[r[1]][r[0]] = 0
                sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}

    nx_graph = nx.from_numpy_array(graph)
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
    nx_digraph = _direct_edges(nx_graph, sepsets)
    nx.relabel_nodes(nx_digraph, lambda i: cols[i], copy=False)
    sepsets = {(cols[k[0]], cols[k[1]]): {'p_val': v['p_val'], 'sepset': [cols[e] for e in v['sepset']]}
            for k, v in sepsets.items()}

    return nx_digraph, sepsets


###########################################################


def run_experiment(seed):
    """Runs the experiment for a given seed and returns results as a dictionary."""

    # Set random seed
    set_random_seed(seed)

    # Reset var_dict before every run to ensure a clean state
    global var_dict
    var_dict = {}  # reset var_dict to avoid contamination between runs    

    path = '../data/custom_mixed_confounding_softplus/'

    # file name adjacency matrix in csv format
    mixed_confounding_adjacency_matrix = 'adj_matrix.csv'
    # file name observational data in csv format
    mixed_confounding_obs_data = 'train.csv'

    # load adjacency matrix and data as pandas dataframe, both files have no header
    adj_matrix = pd.read_csv(path + mixed_confounding_adjacency_matrix, header=None)
    X_df = pd.read_csv(path + mixed_confounding_obs_data, header=None)
    weighted_adj_matrix = pd.read_csv(path + 'W_adj_matrix.csv', header=None)

    B_true = adj_matrix.values
    X = X_df.values
    W_true = weighted_adj_matrix.values

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

    nm_epochs = 1000 # not much happens after 2000 epochs
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

    # %%
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
    GraphDAG(nt.causal_matrix, B_true)

    # calculate accuracy
    nt_met = MetricsDAG(nt.causal_matrix, B_true)
    print(nt_met.metrics)

    ##################################################################################
    # benchmark model 2 pc

    # Peter & Clark (PC)
    # A variant of PC-algorithm, one of [`original`, `stable`, `parallel`]
    pc = PC(variant='parallel', alpha=0.03, ci_test='fisherz', random_state=seed) # F1 of 75%
    #pc = PC(variant='stable', alpha=0.03) # F1 of 66%
    #pc = PC(variant='original', alpha=0.03) # F1 of 55%
    pc.learn(X)

    # plot est_dag and true_dag
    GraphDAG(pc.causal_matrix, B_true)

    # calculate accuracy
    pc_met = MetricsDAG(pc.causal_matrix, B_true)
    print(pc_met.metrics)

    ##################################################################################
    # benchmark model 3 icalingam

    # ICALiNGAM learn
    # max_iter : int, optional (default=1000)
    g_ICAlingam = ICALiNGAM(max_iter=2000, random_state=seed) # F1 of 71%
    g_ICAlingam.learn(X)

    # plot est_dag and true_dag
    GraphDAG(g_ICAlingam.causal_matrix, B_true)

    # calculate accuracy
    g_ICAlingam_met = MetricsDAG(g_ICAlingam.causal_matrix, B_true)
    print(g_ICAlingam_met.metrics)

    ##################################################################################
    # benchmark model 4 lim

    # LiM from lingam library
    LiM = lingam.LiM(w_threshold=0.3, max_iter=200, h_tol=1e-8, rho_max=1e12, lambda1=0.4, seed=seed) # default rho_max is 1e16
    # create array of shape (1, n_features) which indicates which columns/variables in data.values are discrete or continuous variables, where "1" indicates a continuous variable, while "0" a discrete variable.
    is_cont = np.array([(1 if X_df[col].nunique() > 2 else 0) for col in X_df.columns]).reshape(1, -1)
    #LiM.fit(X=X, dis_con=is_cont, only_global=False) # does not finish, even fater 375 minutes
    LiM.fit(X=X, dis_con=is_cont, only_global=True)

    # plot est_dag and true_dag
    LiM_W_est = LiM.adjacency_matrix_
    LiM_B_est = compute_binary_adjacency(LiM_W_est)

    GraphDAG(LiM_B_est, B_true)

    # calculate accuracy
    LiM_met = MetricsDAG(LiM_B_est, B_true)
    print(LiM_met.metrics)
    
    ##################################################################################
    # benchmark model 5 mCMIkNN

    ### parallel pc alg
    # A global dictionary storing the variables passed from the initializer.
    ### parameters
    alpha = 0.05 # significance level, increasing it will increase the number of edges
    processes = 100 # set number of cores
    # set mCMIkNN parameters
    kcmi = 25 # number of nearest neighbors to find for mutual information estimation
    kperm = 5 # neighborhood size for local permutation scheme
    Mperm = 100 # linear reduction in computational cost, total number of permutations to compute for p-value estimation
    #subsample = 1000 # 2 mins
    subsample = 500 # 1 min
    # set maximum level of pcalg (None == infinite)
    max_level = None # limits the maximum size of the conditioning set

    # Create an instance of the mCMIkNN class
    indep_test = mCMIkNN(kcmi=kcmi, kperm=kperm, Mperm=Mperm, subsample=subsample, seed=seed)

    # Run the parallel_stable_pc algorithm
    graph, sepsets = parallel_stable_pc(X_df, indep_test, alpha=alpha, processes=processes, max_level=max_level)

    # plot est_dag and true_dag
    mCMIkNN_B_est = nx.to_numpy_array(graph)

    GraphDAG(mCMIkNN_B_est, B_true)

    # calculate accuracy
    mCMIkNN_met = MetricsDAG(mCMIkNN_B_est, B_true)

    print(mCMIkNN_met.metrics)

    ##################################################### EVALUATION METRICS #####################################################

    ###### PLACE METRIC COMPUTATION CODE HERE ######

    # Define models and their corresponding adjacency/weight matrices
    models = {
        "Our (PC)": (B_est, W_est),
        "Peter Clark": (pc.causal_matrix, None),
        "NOTEARS": (nt.causal_matrix, nt.weight_causal_matrix),
        "ICALiNGAM": (g_ICAlingam.causal_matrix, g_ICAlingam.weight_causal_matrix),
        "LiM": (LiM_B_est, None),
        "mCMIkNN": (mCMIkNN_B_est, None),
    }

    # Indices for treatment (intervention) and target variables
    interv_node_idx = 0
    target_node_idx = 11

    # Initialize dictionary with lists for each metric
    results = {
        "seed": seed,
        "method": [],
        "SHD_absolute": [],
        "SHD_normalized": [],
        "SID_absolute": [],
        "SID_normalized": [],
        "AID_absolute": [],
        "AID_normalized": [],
        "F1_Score": [],
        "TEE": [],
        "AUROC": [],
        "AUPRC": [],
        "True_Total_Effect": [],
        "Estimated_Total_Effect": [],
    }

    # Compute metrics for each method
    for model_name, (B, W) in models.items():
        B_binary = (B > 0).astype(np.int8)  # Convert adjacency to binary

        # Append results for each method
        results["method"].append(model_name)
        results["SHD_absolute"].append(compute_SHD((B_true > 0).astype(np.int8), B_binary)[1])
        results["SHD_normalized"].append(compute_SHD((B_true > 0).astype(np.int8), B_binary)[0])
        results["SID_absolute"].append(compute_SID((B_true > 0).astype(np.int8), B_binary)[1])
        results["SID_normalized"].append(compute_SID((B_true > 0).astype(np.int8), B_binary)[0])
        results["AID_absolute"].append(compute_ancestor_AID((B_true > 0).astype(np.int8), B_binary)[1])
        results["AID_normalized"].append(compute_ancestor_AID((B_true > 0).astype(np.int8), B_binary)[0])
        results["F1_Score"].append(compute_F1_directed(B_true, B))

        # Compute Total Effect Estimation Error (TEE), AUROC, and AUPRC (if weights exist)
        if W is not None:
            true_effect, est_effect, tee = compute_TEE(W_true, W, interv_node_idx, target_node_idx)
            _B_true = (np.abs(W_true) > 0).astype(int)

            results["TEE"].append(tee)
            results["AUROC"].append(compute_AUROC(_B_true, W)[-1])
            results["AUPRC"].append(compute_AUPRC(_B_true, W)[-1])
            results["True_Total_Effect"].append(true_effect)
            results["Estimated_Total_Effect"].append(est_effect)
        else:
            results["TEE"].append(None)
            results["AUROC"].append(None)
            results["AUPRC"].append(None)
            results["True_Total_Effect"].append(None)
            results["Estimated_Total_Effect"].append(None)

    return results


# Function to save results
def save_results(results_list, output_file):
    """Merge results from different runs and save to a JSON file robustly."""
    
    print("📁 Merging and saving results...")

    try:
        # Ensure all values in each dictionary are lists
        processed_results = []
        for res in results_list:
            processed_results.append({key: ([value] if not isinstance(value, list) else value) for key, value in res.items()})

        # Convert processed results into a DataFrame
        df = pd.DataFrame(processed_results)

        # Convert NaN and Inf values to None before saving
        df = df.replace([np.nan, np.inf, -np.inf], None)

        # Save results to JSON
        df.to_json(output_file, orient="records", indent=4)
        print(f"✅ Merged results saved to {output_file}")

    except Exception as e:
        print(f"⚠️ Error saving results: {e}")


if __name__ == "__main__":

    seed_values = [2]

    results_all = [run_experiment(seed) for seed in seed_values]

    # Save merged results
    save_results(results_all, output_file)