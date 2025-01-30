import os
import sys
import itertools
import networkx as nx
import numpy as np
import pandas as pd

# Set up paths
examples_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples'))
sys.path.append(examples_dir)

# Import utility functions
from causal_helpers import set_random_seed
from cyclic_obs_data_generator import sample_ER_dcg, sample_SF_dcg, sample_NWS_dcg
from cyclic_obs_data_generator import sample_W, sample_data

# Set up data directory
data_dir = "/share/amine.mcharrak/cyclic_data"
os.makedirs(data_dir, exist_ok=True)

############################## HELPER FUNCTIONS ##############################

# Function to save dataset
def save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples):
    """
    Save the generated dataset to a directory.
    """
    # Sample weighted adjacency matrix
    W, _ = sample_W(B)

    # Generate synthetic data
    scales = np.ones(d, dtype=float)
    X, prec_matrix = sample_data(W=W, scales=scales, num_samples=num_samples, noise_type=noise_type)

    # Convert to DataFrame
    data = pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])

    # Verify cyclicity
    G = nx.DiGraph(B)
    is_cyclic = not nx.is_directed_acyclic_graph(G)
    print(f"Graph Type: {graph_type}, Nodes: {d}, Edges: {n_edges}, Cyclic: {is_cyclic}")

    # Create directory for storing results
    dataset_dir = os.path.join(
        data_dir, f"{d}{graph_type}{n_edges}_linear_cyclic_{noise_type}_seed_{seed}_n_samples_{num_samples}"
    )
    os.makedirs(dataset_dir, exist_ok=True)

    # Save adjacency matrix
    pd.DataFrame(nx.to_numpy_array(G)).to_csv(f"{dataset_dir}/adj_matrix.csv", header=False, index=False)
    
    # Save generated data
    data.to_csv(f"{dataset_dir}/train.csv", header=False, index=False)

    # Save weighted adjacency matrix
    pd.DataFrame(W).to_csv(f"{dataset_dir}/W_adj_matrix.csv", header=False, index=False)

    # Save precision matrix if applicable
    if prec_matrix is not None:
        pd.DataFrame(prec_matrix).to_csv(f"{dataset_dir}/prec_matrix.csv", header=False, index=False)


############################## MAIN SCRIPT ##################################

# Define parameter ranges
seeds = [1, 2, 3, 4, 5]  # Different seeds for different datasets
num_samples_list = [100, 500, 1000, 2000, 5000]
num_vars_list = [10, 15, 20]  # Number of nodes in the graph
graph_types = ["ER", "SF", "NWS"]  # Graph models
noise_types = ["GAUSS-EV", "SOFTPLUS", "EXP", "UNIFORM"]  # Noise types
p_densities = [0.3, 0.5, 0.7]  # For NWS graphs
e_to_d_ratios = [2, 4, 6]  # For ER and SF graphs
max_cycle = 3  # Max cycle length

# Compute total number of datasets
total_datasets = (
    len(seeds) * len(num_samples_list) * len(num_vars_list) * len(graph_types) * len(noise_types) * len(e_to_d_ratios)
)
print(f"Generating {total_datasets} datasets...")

# Track progress
dataset_count = 0

# Generate all dataset combinations
for seed, num_samples, d, graph_type, noise_type in itertools.product(
    seeds, num_samples_list, num_vars_list, graph_types, noise_types
):
    # Set random seed
    set_random_seed(seed)
    max_degree = d - 1  # Max out-degree per node

    # Generate graphs
    if graph_type == "NWS":
        for p_density in p_densities:
            B = sample_NWS_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=p_density)
            n_edges = int(np.sum(B))  # Ensure correct edge count

            # Generate and store data for each configuration
            dataset_count += 1
            print(f"({dataset_count}/{total_datasets}) Generating: {graph_type}, p_density={p_density}, Noise={noise_type}")

            # Save dataset
            save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples)

    else:  # ER or SF
        for e_to_d_ratio in e_to_d_ratios:
            n_edges = int(e_to_d_ratio * d)  # Initial estimate

            if graph_type == "ER":
                B = sample_ER_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=None, n_edges=n_edges)
            elif graph_type == "SF":
                B = sample_SF_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=None, n_edges=n_edges)

            n_edges = int(np.sum(B))  # Ensure correct edge count after sampling

            # Generate and store data for each configuration
            dataset_count += 1
            print(f"({dataset_count}/{total_datasets}) Generating: {graph_type}, e_to_d_ratio={e_to_d_ratio}, Noise={noise_type}")

            # Save dataset
            save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples)

print("Dataset generation complete!")

