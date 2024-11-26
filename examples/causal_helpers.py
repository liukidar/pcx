import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import jax
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx


def set_random_seed(seed):
    # Set the seed for reproducibility    
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)

# Function to load the adjacency matrix
def load_adjacency_matrix(file_name):
    adj_matrix = np.load(file_name)
    print(f"Adjacency matrix loaded from {file_name}")
    return adj_matrix


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


import numpy as np


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0)), connectome=False):
    """
    Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] adjacency matrix of DAG.
            - If connectome=False, B is a binary adjacency matrix (entries are 0 or 1).
            - If connectome=True, B is a natural number weighted adjacency matrix (positive integers representing strengths).
        w_ranges (tuple): Disjoint weight ranges, e.g., ((-2.0, -0.5), (0.5, 2.0)).
        connectome (bool): If True, B represents strengths, and the sampled parameters will preserve the ordering of these strengths.

    Returns:
        W (np.ndarray): [d, d] weighted adjacency matrix of DAG.
    """
    W = np.zeros(B.shape)
    if not connectome:
        # Original behavior: B is a binary adjacency matrix
        S = np.random.randint(len(w_ranges), size=B.shape)
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
    else:
        # B contains positive integers representing strengths
        # Map B[i,j] to absolute values, preserving ordering
        nonzero_indices = np.where(B != 0)
        B_nonzero = B[nonzero_indices]
        B_min = np.min(B_nonzero)
        B_max = np.max(B_nonzero)
        # Determine the absolute weight range from w_ranges
        abs_w_values = [abs(low) for (low, high) in w_ranges] + [abs(high) for (low, high) in w_ranges]
        abs_w_low = min(abs_w_values)
        abs_w_high = max(abs_w_values)
        # Avoid division by zero if B_min == B_max
        if B_min == B_max:
            target_abs_values = np.full(B_nonzero.shape, abs_w_low)
        else:
            # Linear mapping to [abs_w_low, abs_w_high], preserving ordering
            target_abs_values = ((B_nonzero - B_min) / (B_max - B_min)) * (abs_w_high - abs_w_low) + abs_w_low
        # Randomly select which weight range to use for each edge
        S = np.random.randint(len(w_ranges), size=B_nonzero.shape)
        # Assign weights
        for idx, (i, j) in enumerate(zip(nonzero_indices[0], nonzero_indices[1])):
            range_idx = S[idx]
            low, high = w_ranges[range_idx]
            # Determine sign based on selected range
            if low >= 0 and high > 0:
                sign = 1  # Positive range
            elif low < 0 and high <= 0:
                sign = -1  # Negative range
            else:
                # Handle ranges that cross zero
                sign = np.random.choice([-1, 1])
            # Assign weight with the sign and mapped absolute value
            W[i, j] = sign * target_abs_values[idx]
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_linear_sem_cyclic(W, n, sem_type, noise_scale=None, max_iter=1000, tol=1e-6):
    """Simulate samples from linear SEM with feedback loops (cyclic graph).

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of directed cyclic graph
        n (int): num of samples
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        max_iter (int): maximum number of iterations for convergence
        tol (float): tolerance for convergence

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, w, scale, sem_type):
        """Simulate a single SEM equation for a given node."""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale

    X = np.zeros([n, d])
    prev_X = np.inf * np.ones_like(X)

    for iteration in range(max_iter):
        for j in range(d):
            parents = np.where(W[:, j] != 0)[0]
            X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], sem_type)
        
        # Check for convergence
        if np.max(np.abs(X - prev_X)) < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
        prev_X = X.copy()
    else:
        print("Warning: Maximum iterations reached without convergence.")
    
    return X

def plot_adjacency_matrices(true_matrix, est_matrix, save_path=None):
    """Plot true and estimated adjacency matrices side by side and optionally save the figure."""
    plt.figure(figsize=(20, 8))

    # Plot the estimated adjacency matrix on the left
    plt.subplot(1, 2, 1)
    sns.heatmap(est_matrix, cmap='viridis', cbar=True)
    plt.title('Binary Estimated Graph')
    plt.xlabel('Node')
    plt.ylabel('Node')

    # Plot the true adjacency matrix on the right
    plt.subplot(1, 2, 2)
    sns.heatmap(true_matrix, cmap='viridis', cbar=True)
    plt.title('Binary True Graph')
    plt.xlabel('Node')
    plt.ylabel('Node')

    # Save the figure if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the plot to the specified path

    plt.show()  # Display the combined plot

# Save a NetworkX graph to a file using pickle
def save_graph(graph, file_name):
    """Saves a NetworkX graph to a file using pickle."""
    with open(file_name, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    print(f"Graph saved to {file_name}")

# Load a NetworkX graph from a file using pickle
def load_graph(file_name):
    """Loads a NetworkX graph from a file using pickle."""
    with open(file_name, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {file_name}")
    return graph

# Save the adjacency matrix of a NetworkX graph to a .npy file
def save_adjacency_matrix(graph, file_name):
    """Converts a NetworkX graph to an adjacency matrix and saves it as a .npy file."""
    adj_matrix = nx.to_numpy_array(graph, weight='weight')
    np.save(file_name, adj_matrix)  # Save without adding .npy manually
    print(f"Adjacency matrix saved to {file_name}.npy")

# Load the adjacency matrix from a .npy file
def load_adjacency_matrix(file_name):
    """Loads an adjacency matrix from a .npy file"""
    adj_matrix = np.load(file_name)
    print(f"Adjacency matrix loaded from {file_name}")
    return adj_matrix

# Example usage:
if __name__ == "__main__":
    # For connectome=False (original behavior)
    B_binary = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    W_binary = simulate_parameter(B_binary)
    print("Weighted adjacency matrix (connectome=False):\n", W_binary)

    # For connectome=True
    B_strengths = np.array([
        [0, 10, 0],
        [0, 0, 20],
        [30, 0, 0]
    ])
    W_strengths = simulate_parameter(B_strengths, connectome=True)
    print("Weighted adjacency matrix (connectome=True):\n", W_strengths)
    print("Absolute weights:\n", np.abs(W_strengths))