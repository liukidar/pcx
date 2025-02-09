import os
import sys
import itertools
import networkx as nx
import numpy as np
import random
import pandas as pd
import concurrent.futures

# Set up paths
examples_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples'))
sys.path.append(examples_dir)

# Import utility functions
from mixed_obs_data_generator import sample_B, sample_W, simulate_linear_mixed_sem, is_DAG

# dglearn
from dglearn.dglearn.dg.adjacency_structure import AdjacencyStucture
from dglearn.dglearn.dg.graph_equivalence_search import GraphEquivalenceSearch

# Set up data directory
data_dir = "/share/amine.mcharrak/mixed_data_final"
os.makedirs(data_dir, exist_ok=True)

############################## HELPER FUNCTIONS ##############################

# Function to set random seed
def set_random_seed(seed):
    """Set the seed for NumPy and Python random functions, avoiding JAX."""
    np.random.seed(seed)
    random.seed(seed)

# Function to generate dataset
def generate_dataset(i, total_datasets, seed, d, graph_type, e_to_d_ratio, num_samples_list):
    """
    Generate dataset for given parameters and save it efficiently.
    """
    set_random_seed(seed)

    n_edges = e_to_d_ratio * d  # Number of edges
    B = sample_B(d, n_edges, graph_type)
    W = sample_W(B)

    # Verify cyclicity
    G = nx.DiGraph(B)
    is_DAG = nx.is_directed_acyclic_graph(G)
    print(f"({i}/{total_datasets}) Generating dataset with seed {seed}: Graph Type: {graph_type}, Nodes: {d}, Edges: {n_edges}, is_DAG: {is_DAG}")

    # ✅ Store graph structure **once** in a base folder
    base_dir = os.path.join(data_dir, f"{d}{graph_type}{n_edges}_linear_mixed_seed_{seed}")
    os.makedirs(base_dir, exist_ok=True)

    # Save adjacency matrix **once**
    pd.DataFrame(nx.to_numpy_array(G)).to_csv(f"{base_dir}/adj_matrix.csv", header=False, index=False)

    # Save weighted adjacency matrix **once**
    pd.DataFrame(W).to_csv(f"{base_dir}/W_adj_matrix.csv", header=False, index=False)

    # ✅ Generate and store datasets for different `n_disc` and `n_samples`
    for n_disc in range(1, d):  # Iterate over discrete variables
        for n_samples in num_samples_list:  # Iterate over sample sizes
            # Create ordered discrete variable list
            is_cont = np.ones((1, d))  # 1 = continuous
            is_cont[:, :n_disc] = 0   # First `n_disc` are discrete

            # Generate dataset
            X = simulate_linear_mixed_sem(W, n_samples, is_cont=is_cont, sem_type="mixed_random_i_dis")

            # ✅ Save datasets in a structured way
            sub_dir = os.path.join(base_dir, f"n_disc_{n_disc}")  # Subfolder for `n_disc`
            os.makedirs(sub_dir, exist_ok=True)

            # Store `train.csv` for this specific `n_samples`
            pd.DataFrame(X).to_csv(f"{sub_dir}/train_n_samples_{n_samples}.csv", header=False, index=False)

    print(f"({i}/{total_datasets}) Finished dataset with seed {seed}: {base_dir}")


# **Efficient Execution Setup**
seeds = [1, 2, 3, 4, 5]
num_samples_list = [100, 500, 1000, 2000, 5000]  # Different sample sizes
num_vars_list = [10, 15, 20]  # Number of nodes `d`
graph_types = ["ER", "SF", "WS"]  # Graph models
e_to_d_ratios = [1, 2, 3, 4]  # Edge-to-node ratio

# Generate parameter combinations **only once per graph**
parameter_combinations = [
    (seed, d, graph_type, e_to_d_ratio, num_samples_list)
    for seed, d, graph_type, e_to_d_ratio in itertools.product(seeds, num_vars_list, graph_types, e_to_d_ratios)
]

total_datasets = len(parameter_combinations)
print(f"Generating {total_datasets} datasets...")

for i, params in enumerate(parameter_combinations, start=1):
    generate_dataset(i, total_datasets, *params)

print("✅ All datasets generated successfully!")