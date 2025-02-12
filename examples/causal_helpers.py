import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import networkx as nx
import random
import jax
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as jax_numpy_linalg # for expm()
import jax.scipy.linalg as jax_scipy_linalg # for slogdet()
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch


# Function to set random seed in JAX, NumPy, PyTorch, and Python's random module
def set_random_seed(seed):
    """Set random seed for reproducibility across Python, NumPy, JAX, and PyTorch."""

    # 🔹 Set Python & NumPy seed
    random.seed(seed)
    np.random.seed(seed)

    # 🔹 Set JAX PRNGKey properly (ensuring 64-bit precision)
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(seed)  # Correct way to seed JAX

    # 🔹 Set PyTorch seed **only on the main process** to avoid multiprocessing issues
    torch.manual_seed(seed)

    # 🔹 Set CUDA seed **only if available and only on the main process**
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = False  # allow non-deterministic cuDNN functions
        torch.backends.cudnn.benchmark = True  # select fastest kernel

    return key  # Return JAX PRNGKey if needed

############################ JAX Utility and Metric Functions ############################

@jit
def MAE(W_true, W):
    """This function returns the Mean Absolute Error for the difference between the true weighted adjacency matrix W_true and th estimated one, W."""
    MAE_ = jnp.mean(jnp.abs(W - W_true))
    return MAE_

@jax.jit
def compute_h_reg(W):
    """
    Compute the DAG constraint using the exponential trace-based acyclicity constraint (NOTEARS).

    This function calculates the value of the acyclicity constraint for a given
    adjacency matrix using the formulation:
    h_reg = trace(exp(W ⊙ W)) - d
    where ⊙ represents the Hadamard (element-wise) product.

    Parameters
    ----------
    W : jnp.ndarray
        (d, d) adjacency matrix.

    Returns
    -------
    h_reg : float
        The value of the DAG constraint.
    """
    # Dimensions of W
    d = W.shape[0]

    # Compute h_reg using the trace of the matrix exponential
    h_reg = jnp.trace(jax_scipy_linalg.expm(jnp.multiply(W, W))) - d

    return h_reg

@jax.jit
def notears_dag_constraint(W):
    """
    Compute the NOTEARS DAG constraint using the exponential trace-based acyclicity constraint.

    This function calculates the value of the acyclicity constraint for a given
    adjacency matrix using the formulation:
    h_reg = trace(exp(W ⊙ W)) - d
    where ⊙ represents the Hadamard (element-wise) product.

    Parameters
    ----------
    W : jnp.ndarray
        (d, d) adjacency matrix.

    Returns
    -------
    h_reg : float
        The value of the DAG constraint.
    """
    # Dimensions of W
    d = W.shape[0]

    # Compute h_reg using the trace of the matrix exponential
    h_reg = jnp.trace(jax_scipy_linalg.expm(jnp.multiply(W, W))) - d

    return h_reg

@jax.jit
def dagma_dag_constraint(W, s=1.0):
    """
    Compute the DAG constraint using the logdet acyclicity constraint from DAGMA.
    This function is JAX-jitted for improved performance.

    Parameters
    ----------
    W : jnp.ndarray
        (d, d) adjacency matrix.
    s : float, optional
        Controls the domain of M-matrices. Defaults to 1.0.

    Returns
    -------
    h_reg : float
        The value of the DAG constraint.
    """
    # Dimensions of W
    d = W.shape[0]

    # Compute M-matrix for the logdet constraint
    M = s * jnp.eye(d) - jnp.multiply(W, W)

    # Compute the value of the logdet DAG constraint
    h_reg = -jax_numpy_linalg.slogdet(M)[1] + d * jnp.log(s)

    return h_reg


# # Define fucntion to compute h_reg based W with h_reg = jnp.trace(jax.scipy.linalg.expm(W * W)) - d, here * denotes the hadamard product
# def compute_h_reg(W):
#     """This function computes the h_reg term based on the matrix W."""
#     h_reg = jnp.trace(jax.scipy.linalg.expm(W * W)) - W.shape[0]
#     return h_reg

def compute_binary_adjacency(W, threshold=0.3):
    """
    Compute the binary adjacency matrix by thresholding the input matrix.

    Args:
    - W (array-like): The weighted adjacency matrix (can be a JAX array or a NumPy array).
    - threshold (float): The threshold value to determine the binary matrix. Default is 0.3.

    Returns:
    - B_est (np.ndarray): The binary adjacency matrix where each element is True if the corresponding 
                          element in W is greater than the threshold, otherwise False.
    """
    # Convert JAX array to NumPy array if necessary
    if isinstance(W, jnp.ndarray):
        W = np.array(W)

    # Compute the binary adjacency matrix
    B_est = np.array(np.abs(W) > threshold, dtype=int)

    return B_est

###################################### DAG Utility Functions ######################################

def enforce_dag(B):
    """Ensure B is a DAG by removing minimal edges to break cycles."""
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)

    try:
        while True:
            cycle_edges = nx.find_cycle(G, orientation="original")  # Find a cycle
            edge_to_remove = cycle_edges[0]  # Pick the first edge in the cycle
            u, v = edge_to_remove[:2]  # Extract edge nodes
            print(f"🚨 Removing edge {u} → {v} to break cycle")
            G.remove_edge(u, v)  # Remove the edge to break the cycle
    except nx.NetworkXNoCycle:
        pass  # No cycles left, we're done!

    return nx.to_numpy_array(G)

def ensure_DAG(W):
    """
    Ensure that the weighted adjacency matrix corresponds to a DAG.

    Inputs:
        W: numpy.ndarray - a weighted adjacency matrix representing a directed graph

    Outputs:
        W: numpy.ndarray - a weighted adjacency matrix without cycles (DAG)
    """
    # Convert the adjacency matrix to a directed graph
    g = nx.DiGraph(W)

    # Make a copy of the graph to modify
    gg = g.copy()

    # Remove cycles by removing edges
    while not nx.is_directed_acyclic_graph(gg):
        h = gg.copy()

        # Remove all the sources and sinks
        while True:
            finished = True

            for node, in_degree in nx.in_degree_centrality(h).items():
                if in_degree == 0:
                    h.remove_node(node)
                    finished = False

            for node, out_degree in nx.out_degree_centrality(h).items():
                if out_degree == 0:
                    h.remove_node(node)
                    finished = False

            if finished:
                break

        # Find a cycle with a random walk starting at a random node
        node = list(h.nodes)[0]
        cycle = [node]
        while True:
            edges = list(h.out_edges(node))
            _, node = edges[np.random.choice(len(edges))]

            if node in cycle:
                break

            cycle.append(node)

        # Extract the cycle path and adjust it to start at the first occurrence of the repeated node
        cycle = np.array(cycle)
        i = np.argwhere(cycle == node)[0][0]
        cycle = cycle[i:]
        cycle = cycle.tolist() + [node]

        # Find edges in that cycle
        edges = list(zip(cycle[:-1], cycle[1:]))

        # Randomly pick an edge to remove
        edge = edges[np.random.choice(len(edges))]
        gg.remove_edge(*edge)

    # Convert the modified graph back to a weighted adjacency matrix
    W_acyclic = nx.to_numpy_array(gg)

    return W_acyclic

def is_dag_ig(adjacency_matrix: np.ndarray) -> bool:
    """
    Checks if a given adjacency matrix represents a Directed Acyclic Graph (DAG)
    using the igraph library.  Works with both weighted and binary adjacency matrices.

    Parameters:
        adjacency_matrix: A square NumPy array representing the adjacency of a
                          directed graph.  Non-zero entries indicate edges.
                          Can be weighted or binary.

    Returns:
        True if the graph is a DAG, False otherwise.

    Raises:
        TypeError: If adjacency_matrix is not a NumPy array.
        ValueError: If adjacency_matrix is not a square 2D array.

    Examples:
        >>> import numpy as np
        >>> import igraph as ig
        >>> # DAG (acyclic)
        >>> W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> is_dag_ig(W)
        True
        >>> # Cyclic graph
        >>> W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        >>> is_dag_ig(W)
        False
    """
    if not isinstance(adjacency_matrix, np.ndarray):
        raise TypeError("adjacency_matrix must be a NumPy array")
    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be a square 2D array")

    # Convert to list of lists (igraph requirement for weighted adjacency)
    matrix_list = adjacency_matrix.tolist()
    G = ig.Graph.Weighted_Adjacency(matrix_list)  # Handles both weighted and binary
    return G.is_dag()

def is_dag_nx(adjacency_matrix: np.ndarray) -> bool:
    """
    Checks if a given adjacency matrix represents a Directed Acyclic Graph (DAG)
    using the networkx library. Works with both weighted and binary adjacency matrices.

    Parameters:
        adjacency_matrix: A square NumPy array representing the adjacency of a
                          directed graph. Non-zero entries indicate edges.
                          Can be weighted or binary.

    Returns:
        True if the graph is a DAG, False otherwise.

    Raises:
        TypeError: If adjacency_matrix is not a NumPy array.
        ValueError: If adjacency_matrix is not a square 2D array.
    
    Examples:
        >>> import numpy as np
        >>> import networkx as nx
        >>> # DAG (acyclic)
        >>> W = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        >>> is_dag_nx(W)
        True
        >>> # Cyclic graph
        >>> W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        >>> is_dag_nx(W)
        False

    """
    if not isinstance(adjacency_matrix, np.ndarray):
        raise TypeError("adjacency_matrix must be a NumPy array")
    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be a square 2D array")

    graph = nx.DiGraph(adjacency_matrix)  # Handles both weighted and binary
    return nx.is_directed_acyclic_graph(graph)

def pick_source_target(B_true, random_state=None):
    """
    Given a true adjacency matrix B_true (binary or weighted, where a nonzero entry
    indicates an edge from i to j), returns a tuple (source, target) such that target
    is a descendant of source.

    Parameters
    ----------
    B_true : numpy.ndarray
        Adjacency matrix of shape (n, n).
    random_state : int or np.random.Generator, optional
        If an int is provided, a new Generator is created with that seed.
        Otherwise, if a Generator is provided, it is used. Default uses a new generator.

    Returns
    -------
    source : int
        Index of the source (intervention) node.
    target : int
        Index of the target (effect) node, which is a descendant of source.

    Raises
    ------
    ValueError
        If no node with descendants exists.

    # Example: create a simple true adjacency matrix for a DAG
    B_true = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    src, tgt = pick_source_target(B_true, random_state=42)
    print("Source:", src, "Target:", tgt)
    
    """
    # Create a random generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Create a directed graph from the adjacency matrix.
    # Assume B_true[i, j] != 0 indicates an edge from i to j.
    n = B_true.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if B_true[i, j] != 0:
                G.add_edge(i, j)

    # Find candidate source nodes: nodes that have at least one descendant.
    candidate_sources = [node for node in G.nodes() if len(nx.descendants(G, node)) > 0]
    if not candidate_sources:
        raise ValueError("No node with descendants found in the graph.")

    source = rng.choice(candidate_sources)
    descendants = list(nx.descendants(G, source))
    target = rng.choice(descendants)
    return source, target

####################### Graph and Data Simulation #######################

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
    if not is_dag_nx(W):
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


################################ PLotting and other analysing functions ################################

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

def plot_weights_distribution(W_est, bins=np.arange(0, 2.1, 0.1), title="Histogram of Absolute Values in Weight Matrix"):
    """
    Plots a histogram of the absolute values in the weight matrix W_est.

    Parameters:
    - W_est (np.ndarray): Estimated weight matrix.
    - bins (np.ndarray): Bins for histogram plotting.
    - title (str): Title for the histogram plot.

    Usage Example:
    ```python
    plot_weights_distribution(W_est)
    ```
    """
    abs_values = np.abs(W_est).flatten()

    plt.hist(abs_values, bins=bins, edgecolor='black')
    plt.xlabel('Absolute Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.show()

def analyze_weight_thresholds(W_est, B_est, threshold_values=[0.1, 0.2, 0.3]):
    """
    Analyzes the weight matrix W_est by counting values exceeding given thresholds
    and comparing them to the number of edges in the adjacency matrix B_est.

    Parameters:
    - W_est (np.ndarray): Estimated weight matrix.
    - B_est (np.ndarray): Binary adjacency matrix indicating edges.
    - threshold_values (list of float): Thresholds for counting significant absolute values.

    Returns:
    - counts (dict): Dictionary mapping each threshold to its count of exceeding values.
    - num_edges (int): Total number of edges in B_est.

    Usage Example:
    ```python
    counts, num_edges = analyze_weight_thresholds(W_est, B_est, threshold_values=[0.1, 0.2, 0.3])
    ```
    """
    abs_values = np.abs(W_est).flatten()

    counts = {threshold: np.sum(abs_values > threshold) for threshold in threshold_values}

    for threshold, count in counts.items():
        print(f"Number of absolute values in W_est larger than {threshold}: {count}")

    num_edges = np.sum(B_est)
    print("Number of edges in B_est:", num_edges)

    return counts, num_edges

################################ Saving and Loading Graphs ################################

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