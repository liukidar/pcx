import os
# Set up JAX environment to prevent CUDA errors
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Forces JAX to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hides all GPUs from CUDA
import sys
import networkx as nx
import numpy as np
import random
import pandas as pd

import concurrent.futures
import itertools
from joblib import Parallel, delayed

# Set up paths
examples_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples'))
sys.path.append(examples_dir)

# Import utility functions
from cyclic_obs_data_generator import sample_ER_dcg, sample_SF_dcg, sample_NWS_dcg
from cyclic_obs_data_generator import sample_W, sample_data

# Set up data directory
data_dir = "/share/amine.mcharrak/cyclic_data_final"
os.makedirs(data_dir, exist_ok=True)

# dglearn
from dglearn.dglearn.dg.adjacency_structure import AdjacencyStucture
from dglearn.dglearn.dg.graph_equivalence_search import GraphEquivalenceSearch, TooManyVisitedGraphsError

############################## HELPER FUNCTIONS ##############################

# Function to set random seed
def set_random_seed(seed):
    """Set the seed for NumPy and Python random functions, avoiding JAX."""
    np.random.seed(seed)
    random.seed(seed)

# Function to save dataset
def save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples_list):
    """
    Save multiple subsampled datasets from a single generated dataset.
    
    Args:
        data_dir (str): Directory to save the datasets.
        B (np.ndarray): Adjacency matrix of the generated graph.
        d (int): Number of variables (nodes).
        graph_type (str): Type of graph (ER, SF, NWS).
        n_edges (int): Number of edges in the graph.
        noise_type (str): Type of noise model.
        seed (int): Random seed used for generation.
        num_samples_list (list): List of sample sizes to generate.
    """
    # ✅ Always generate a dataset with 5000 samples
    max_samples = max(num_samples_list)  # Ensures we generate the largest dataset needed
    W, _ = sample_W(B)

    # Generate synthetic data once
    scales = np.ones(d, dtype=float)
    X_full, prec_matrix = sample_data(W=W, scales=scales, num_samples=max_samples, noise_type=noise_type)

    # Convert to DataFrame
    X_full_df = pd.DataFrame(X_full, columns=[f"X{i}" for i in range(d)])

    # Verify cyclicity
    G = nx.DiGraph(B)
    is_cyclic = not nx.is_directed_acyclic_graph(G)
    print(f"Saving Graph Type: {graph_type}, Nodes: {d}, Edges: {n_edges}, Cyclic: {is_cyclic}")

    # Create base directory
    base_dir = os.path.join(data_dir, f"{d}{graph_type}{n_edges}_linear_cyclic_{noise_type}_seed_{seed}")
    os.makedirs(base_dir, exist_ok=True)

    # Save adjacency matrix
    pd.DataFrame(nx.to_numpy_array(G)).to_csv(f"{base_dir}/adj_matrix.csv", header=False, index=False)
    
    # Save weighted adjacency matrix
    pd.DataFrame(W).to_csv(f"{base_dir}/W_adj_matrix.csv", header=False, index=False)

    # Save precision matrix
    pd.DataFrame(prec_matrix).to_csv(f"{base_dir}/prec_matrix.csv", header=False, index=False)

    # ✅ Now save different dataset sizes from `X_full_df`
    for num_samples in num_samples_list:
        dataset_dir = os.path.join(base_dir, f"n_samples_{num_samples}")
        os.makedirs(dataset_dir, exist_ok=True)

        # Take the first `num_samples` rows
        X_subset = X_full_df.iloc[:num_samples]

        # Save subset dataset
        X_subset.to_csv(f"{dataset_dir}/train.csv", header=False, index=False)


############################## MAIN SCRIPT ##################################

def generate_dataset(seed, d, graph_type, noise_type, e_to_d_ratio, num_samples_list, 
                     max_cycle=3, max_resample_attempts=10, p_densities=None, k_nws=2, edge_add_prob=0.3):  
    """
    Generate dataset for a given combination of parameters.
    If the graph generation fails, it will skip to the next dataset instead of stopping execution.
    """
    set_random_seed(seed)
    max_degree = d - 1  # Max possible degree per node

    if graph_type == "NWS":
        if p_densities is None:
            raise ValueError("p_densities must be provided for NWS graph generation.")

        for p_density in p_densities:
            attempts = 0
            while attempts < max_resample_attempts:
                try:
                    B = sample_NWS_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, 
                                       k=k_nws, p=p_density, edge_add_prob=edge_add_prob)
                    n_edges = int(np.sum(B))

                    G_true = nx.DiGraph(B)
                    edges_list = list(G_true.edges())

                    true_graph = AdjacencyStucture(n_vars=d, edge_list=edges_list)
                    search = GraphEquivalenceSearch(true_graph, report_too_large_class=True)
                    search.search_dfs()
                    break  # ✅ Valid graph found, exit loop
                except TooManyVisitedGraphsError:
                    print(f"⚠️ Resampling B for {graph_type} with p_density={p_density} due to large equivalence class.")
                    attempts += 1
                except ValueError as e:
                    print(f"❌ Graph generation failed: {e}, retrying...")
                    attempts += 1

            if attempts == max_resample_attempts:
                print(f"❌ Max resampling attempts reached for {graph_type}, skipping this dataset.")
                return
        
        # ✅ Move save_dataset **outside** of the for-loop
        save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples_list)

    else:  # ER or SF
        attempts = 0
        while attempts < max_resample_attempts:
            try:
                n_edges = int(e_to_d_ratio * d)  # ✅ Use fixed edge-to-node ratio

                if graph_type == "ER":
                    B = sample_ER_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, 
                                      n_edges=n_edges, edge_add_prob=edge_add_prob)  # ✅ Pass edge_add_prob
                elif graph_type == "SF":
                    B = sample_SF_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, 
                                      n_edges=n_edges, edge_add_prob=edge_add_prob)  # ✅ Pass edge_add_prob
                else:
                    raise ValueError(f"Invalid graph type: {graph_type}. Choose from 'ER', 'SF', or 'NWS'.")

                n_edges = int(np.sum(B))

                G_true = nx.DiGraph(B)
                edges_list = list(G_true.edges())

                true_graph = AdjacencyStucture(n_vars=d, edge_list=edges_list)
                search = GraphEquivalenceSearch(true_graph, report_too_large_class=True)
                search.search_dfs()
                break  # ✅ Valid graph found, exit loop
            except TooManyVisitedGraphsError:
                print(f"⚠️ Resampling B for {graph_type} with e_to_d_ratio={e_to_d_ratio} due to large equivalence class.")
                attempts += 1
            except ValueError as e:
                print(f"❌ Graph generation failed: {e}, retrying...")
                attempts += 1

        if attempts == max_resample_attempts:
            print(f"❌ Max resampling attempts reached for {graph_type}, skipping this dataset.")
            return

    # ✅ Move save_dataset **outside** of the while-loop
    save_dataset(data_dir, B, d, graph_type, n_edges, noise_type, seed, num_samples_list)

# # Define parameter ranges
# #seeds = [1, 2, 3, 4, 5]
# seeds = [1]  
# #num_samples_list = [500, 1000, 2000, 5000]
# num_samples_list = [500]
# num_vars_list = [10, 15, 20]
# graph_types = ["ER", "SF", "NWS"]
# noise_types = ["GAUSS-EV"]
# #p_densities = [0.1, 0.3, 0.5]  # Only used for NWS
# p_densities = [0.1, 0.3]  # Only used for NWS
# #e_to_d_ratios = [1, 2, 3, 4]  # Used for ER and SF
# e_to_d_ratios = [1, 2]  # Used for ER and SF
# max_cycle = 3  

# # Compute total number of datasets (now includes e_to_d_ratios)
# total_datasets = (
#     len(seeds) * len(num_samples_list) * len(num_vars_list) * len(graph_types) * len(noise_types) * len(e_to_d_ratios)
# )
# print(f"Generating {total_datasets} datasets...")

# # Generate all parameter combinations, including e_to_d_ratios
# parameter_combinations = list(itertools.product(seeds, num_samples_list, num_vars_list, graph_types, noise_types, e_to_d_ratios))

# # Parallel processing using joblib
# num_workers = min(64, len(parameter_combinations))  # Use up to 64 workers, but no more than the number of tasks

# Parallel(n_jobs=num_workers)(
#     delayed(generate_dataset)(seed, num_samples, d, graph_type, noise_type, e_to_d_ratio)  
#     for seed, num_samples, d, graph_type, noise_type, e_to_d_ratio in parameter_combinations
# )

# print("✅ Dataset generation complete!")

# Define parameter ranges
seeds = [1]
num_samples_list = [500, 1000, 2000, 5000]  # ✅ Passed as a list
num_vars_list = [10, 15, 20]
#graph_types = ["ER", "SF", "NWS"]
graph_types = ["ER", "SF"]
#graph_types = ["NWS"]
noise_types = ["GAUSS-EV"]
p_densities = [0.1, 0.4, 0.7]  # Used only for NWS
e_to_d_ratios = [1, 2, 3]  # Used for ER and SF
max_cycle = 3
k_nws = 3
edge_add_prob = 0.1

# Compute total number of datasets (only counting graph generation cases)
total_graphs = len(seeds) * len(num_vars_list) * len(graph_types) * len(noise_types) * len(e_to_d_ratios)
total_datasets = total_graphs * len(num_samples_list)
print(f"Generating {total_datasets} datasets across {total_graphs} unique graphs...")

# Generate all parameter combinations for **graph generation** (B)
parameter_combinations = list(itertools.product(seeds, num_vars_list, graph_types, noise_types, e_to_d_ratios))

# Run in parallel
num_workers = min(100, len(parameter_combinations))  
Parallel(n_jobs=num_workers)(
    delayed(generate_dataset)(
        seed=seed, 
        d=d, 
        graph_type=graph_type, 
        noise_type=noise_type, 
        e_to_d_ratio=e_to_d_ratio, 
        num_samples_list=num_samples_list,  
        max_cycle=max_cycle, 
        p_densities=p_densities if graph_type == "NWS" else None,  
        k_nws=k_nws if graph_type == "NWS" else None,  
        edge_add_prob=edge_add_prob  
    )  
    for seed, d, graph_type, noise_type, e_to_d_ratio in parameter_combinations
)

print("✅ Dataset generation complete!")