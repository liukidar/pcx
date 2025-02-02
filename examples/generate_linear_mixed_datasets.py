import os
import sys
import itertools
import networkx as nx
import numpy as np
import random
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Set up paths
examples_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples'))
sys.path.append(examples_dir)

# Import utility functions
from mixed_obs_data_generator import sample_B, sample_W, simulate_linear_mixed_sem, is_DAG

# dglearn
from dglearn.dglearn.dg.adjacency_structure import AdjacencyStucture
from dglearn.dglearn.dg.graph_equivalence_search import GraphEquivalenceSearch

# Set up data directory
data_dir = "/share/amine.mcharrak/mixed_data"
os.makedirs(data_dir, exist_ok=True)

# Parameter combinations
seeds = [1, 2, 3, 4, 5]
num_samples_list = [100, 500, 1000, 2000, 5000]
num_vars_list = [10, 15, 20]  # Number of nodes `d`
graph_types = ["ER", "SF", "WS"]  # Graph models
e_to_d_ratios = [2, 4, 6]  # Edge-to-node ratio, this determines number of edges

############################## HELPER FUNCTIONS ##############################

# Function to set random seed
def set_random_seed(seed):
    """Set the seed for NumPy and Python random functions, avoiding JAX."""
    np.random.seed(seed)
    random.seed(seed)

# Function to generate dataset
def generate_dataset(seed, n_samples, d, graph_type, e_to_d_ratio):
    """Generate dataset for given parameters and save it to disk."""
    random.seed(seed)
    np.random.seed(seed)

    n_edges = e_to_d_ratio * d  # Number of edges in the graph
    B = sample_B(d, n_edges, graph_type)

    # Verify cyclicity
    G = nx.DiGraph(B)
    is_DAG = nx.is_directed_acyclic_graph(G)
    print(f"Graph Type: {graph_type}, Nodes: {d}, Edges: {n_edges}, is_DAG: {is_DAG}")

    W = sample_W(B)

    # Create dataset directory for this configuration
    dataset_dir = os.path.join(data_dir, f"{d}{graph_type}{n_edges}_linear_mixed_seed_{seed}_n_samples_{n_samples}")
    os.makedirs(dataset_dir, exist_ok=True)

    for n_disc in range(1, d):  # Iterate n_disc from 1 to d-1
        # Create ordered discrete variable list
        is_cont = np.ones(d)  # All continuous (1)
        is_cont[:n_disc] = 0   # Set first `n_disc` variables to discrete (0)

        is_cont = np.ones((1, d))  # Ensure 2D array shape (1, d), all continuous (b/c of 1)
        is_cont[:, :n_disc] = 0   # Set first `n_disc` variables to discrete (via 0)

        X = simulate_linear_mixed_sem(W, n_samples, is_cont=is_cont , sem_type="mixed_random_i_dis")

        # Save data
        sub_dir = os.path.join(dataset_dir, f"n_disc_{n_disc}")
        os.makedirs(sub_dir, exist_ok=True)

        # Save adjacency matrix
        pd.DataFrame(nx.to_numpy_array(G)).to_csv(f"{sub_dir}/adj_matrix.csv", header=False, index=False)

        # Save generated data
        pd.DataFrame(X).to_csv(f"{sub_dir}/train.csv", header=False, index=False)

        # Save weighted adjacency matrix
        pd.DataFrame(W).to_csv(f"{sub_dir}/W_adj_matrix.csv", header=False, index=False)

    print(f"Finished: {dataset_dir}")

# **Parallel Processing Setup**
parameter_combinations = list(itertools.product(seeds, num_samples_list, num_vars_list, graph_types, e_to_d_ratios))

num_workers = min(200, len(parameter_combinations))  # Limit to 200 workers or available tasks
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    executor.map(lambda args: generate_dataset(*args), parameter_combinations)

print("✅ All datasets generated successfully!")

# TODO: Add some status/progress tracking