import sys
import os
import networkx as nx

print("Current Working Directory:", os.getcwd())

# Add the examples directory to sys.path so we can import the set_random_seed function and other utilities from causal_helpers
examples_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples'))
sys.path.append(examples_dir)
# Now import set_random_seed directly from causal_helpers
from causal_helpers import set_random_seed
from cyclic_obs_data_generator import sample_ER_dcg, sample_SF_dcg, sample_NWS_dcg
from cyclic_obs_data_generator import sample_W, sample_data

# Add the data directory to sys.path so we can save and load data files
data_dir = os.path.abspath(os.path.join(examples_dir, '..', 'data'))

seed = 1 # to be sampled from [1, 2, 3, 4, 5]
set_random_seed(seed)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Current Working Directory:", os.getcwd())

# Example usage
num_samples = 5000 # number of samples to be sampled from [100, 500, 1000, 2000, 5000]
d = 10 # number of variables in the graph to be sampled from [10, 15, 20]
#max_degree = int(d // 2) # ensure that no node has more than half the number of nodes as neighbors
max_degree = d - 1 # ensure that no node has more than d - 1 neighbors
max_cycle = 3 # maximum cycle length in the graph
p_density = 0.3 # density of the graph to be sampled from [0.3,0.5,0.7] uese these for nws only
e_to_d_ratio = 2 # ratio of edges to nodes to be sampled from [2,4,6] use these for er and sf only

n_edges = e_to_d_ratio * d # number of edges in the graph

# SELECT 1 NOISE TYPE to be sampled from ["GAUSS-EV", "SOFTPLUS", "EXP", "UNIFORM"]
noise_type = "GAUSS-EV"
#noise_type = "SOFTPLUS"
#noise_type = "EXP"
#noise_type = "UNIFORM"


# SELECT 1 GRAPH TYPE to be sampled from ["ER", "SF", "NWS"]
#graph_type = "SF"
graph_type = "ER"
#graph_type = "NWS"

# Generate data

#B = sample_ER_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=None, n_edges=n_edges)
#B = sample_SF_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=None, n_edges=n_edges)
B = sample_NWS_dcg(d=d, max_degree=max_degree, max_cycle=max_cycle, p=p_density)
print("B created")

#W, noise_scales = sample_W(B)
W, _ = sample_W(B)
print("W created")

scales = np.ones(d, dtype=float)
X, prec_matrix = sample_data(W=W, scales=scales, num_samples=num_samples, noise_type=noise_type)
print("X created")

data = pd.DataFrame(X, columns=[f"X{i}" for i in range(d)])

# Display the adjacency matrix and a preview of the data
print("Adjacency Matrix:")
print(B)
print("\nGenerated Data:")
print(data.head())
print("\nTrue Weights:")
print(W)

G = nx.DiGraph(B)

print("Is the graph directed?", G.is_directed())

is_cyclic = False  # Initialize is_cyclic to False
try:
    cycles = nx.find_cycle(G)
    is_cyclic = True  # Set to True if a cycle is found
    print("Is the graph cyclic?", is_cyclic)
    print(f"There are {len(list(nx.simple_cycles(G)))} cycles in the graph, including: {cycles} for example.")
except nx.NetworkXNoCycle:
    print("Is the graph cyclic?", is_cyclic)  # Use the boolean variable directly

print(f"Number of edges: {G.number_of_edges()}")
print(f"Number of nodes: {G.number_of_nodes()}")

# Save all the data to a directory

# in the end we only store n_vars and n_edges in the directory name because we are interested about the density of the graph using the notation 10ER2, 10SF4, 10NSW3 and so on
dir_name = os.path.join(
    data_dir,
    f"linear_cyclic_{noise_type}_{graph_type}_nvars_{d}_n_edges_{G.number_of_edges()}_max_degree_{max_degree}_max_cycle_{max_cycle}_seed_{seed}_n_samples_{num_samples}"
)

os.makedirs(dir_name, exist_ok=True)

# store the adjacency matrix as a csv file named "adj_matrix.csv" without header
adj_matrix_df = pd.DataFrame(B)
adj_matrix_df.to_csv(f"{dir_name}/adj_matrix.csv", header=False, index=False)

# store the data as a csv file named "train.csv" without header
data.to_csv(f"{dir_name}/train.csv", header=False, index=False)

# store the weighted adjacency matrix as a csv file named "W.csv" without header
W_df = pd.DataFrame(W)
W_df.to_csv(f"{dir_name}/W_adj_matrix.csv", header=False, index=False)

# store the precision matrix as a csv file named "prec_matrix.csv" without header
if prec_matrix is not None:
    prec_matrix_df = pd.DataFrame(prec_matrix)
    prec_matrix_df.to_csv(f"{dir_name}/prec_matrix.csv", header=False, index=False)
