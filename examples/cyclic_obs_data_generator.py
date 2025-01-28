# sampling observational data from cyclic graphs

import numpy as np
import networkx as nx


########################## graph sampling functions ##########################

# ER cycle graph sampling
def sample_er_dcg(n_vars, max_degree, max_cycle, p=0.3):
    """
    Sample a Directed Cyclic Graph (DCG) with an Erdős-Rényi degree distribution.

    Args:
        n_vars (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (number of edges) per node.
        max_cycle (int): Maximum allowed length of any cycle in the graph.
        p (float): Probability of edge creation in the Erdős-Rényi graph.

    Returns:
        np.ndarray: Adjacency matrix representing the graph structure.
    """
    while True:  # Repeat until a cyclic graph is generated
        def degree(var):
            """Calculate the degree of a node in the adjacency matrix."""
            return len(np.where(B[:, var])[0]) + len(np.where(B[var, :])[0])

        def reachable(end, start):
            """Check if 'end' is reachable from 'start' using DFS."""
            seen = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node == end:
                    return True
                if node not in seen:
                    seen.add(node)
                    neighbors = np.where(B[node, :])[0]
                    stack.extend(neighbors)
            return False

        def find_scc(var):
            """Find variables in the strongly connected component (SCC) of 'var'."""
            scc = set([var])
            for i in range(B.shape[0]):
                if i != var and reachable(var, i) and reachable(i, var):
                    scc.add(i)
            return scc

        # Generate an Erdős-Rényi directed acyclic graph
        G = nx.erdos_renyi_graph(n_vars, p, directed=True)
        G = nx.DiGraph([(u, v) for u, v in G.edges() if u < v])  # Ensure DAG property

        # Add any missing nodes to ensure the graph has exactly n_vars nodes
        for node in range(n_vars):
            if node not in G.nodes:
                G.add_node(node)

        # Convert to adjacency matrix
        B = nx.to_numpy_array(G)

        # Add cycles randomly while respecting constraints
        for i in range(n_vars):
            for j in np.random.permutation(n_vars):
                if i == j or B[i, j] == 1 or B[j, i] == 1:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                direction = np.random.choice([True, False])
                if direction:
                    B[i, j] = 1
                else:
                    B[j, i] = 1
                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    if direction:
                        B[i, j] = 0
                    else:
                        B[j, i] = 0

        # Validate degrees
        for i in range(n_vars):
            assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

        # Check if the graph is cyclic using NetworkX
        G = nx.DiGraph(B)
        is_cyclic = not nx.is_directed_acyclic_graph(G)
        if is_cyclic:
            return B

# SF cycle graph sampling
def sample_sf_dcg(n_vars, max_degree, max_cycle):
    """
    Sample a Directed Cyclic Graph (DCG) with a scale-free degree distribution.

    Args:
        n_vars (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (number of edges) per node.
        max_cycle (int): Maximum allowed length of any cycle in the graph.

    Returns:
        np.ndarray: Adjacency matrix representing the graph structure.
    """
    while True:  # Repeat until a cyclic graph is generated
        B = np.zeros((n_vars, n_vars))

        def degree(var):
            return len(np.where(B[:, var])[0]) + len(np.where(B[var, :])[0])

        def reachable(end, start):
            """Check if 'end' is reachable from 'start' using DFS."""
            seen = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node == end:
                    return True
                if node not in seen:
                    seen.add(node)
                    neighbors = np.where(B[node, :])[0]
                    stack.extend(neighbors)
            return False

        def find_scc(var):
            """Find variables in the strongly connected component (SCC) of 'var'."""
            scc = set([var])
            for i in range(B.shape[0]):
                if i != var and reachable(var, i) and reachable(i, var):
                    scc.add(i)
            return scc

        # Generate a scale-free graph using NetworkX's scale_free_graph
        G = nx.scale_free_graph(n_vars)
        G = nx.DiGraph(G)  # Ensure it is directed

        # Convert to adjacency matrix
        B = nx.to_numpy_array(G)

        # Add cycles randomly while respecting constraints
        for i in range(n_vars):
            for j in np.random.permutation(n_vars):
                if i == j or B[i, j] == 1 or B[j, i] == 1:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                direction = np.random.choice([True, False])
                if direction:
                    B[i, j] = 1
                else:
                    B[j, i] = 1
                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    if direction:
                        B[i, j] = 0
                    else:
                        B[j, i] = 0

        # Validate degrees
        for i in range(n_vars):
            assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

        # Check if the graph is cyclic using NetworkX
        G = nx.DiGraph(B)
        is_cyclic = not nx.is_directed_acyclic_graph(G)
        if is_cyclic:
            return B

# NWS cycle graph sampling     
import numpy as np
import networkx as nx

def sample_nws_dcg(n_vars, max_degree, max_cycle, k=2, p=0.5):
    """
    Sample a Directed Cyclic Graph (DCG) using the Newman-Watts-Strogatz model.

    Args:
        n_vars (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (in + out) per node.
        max_cycle (int): Maximum allowed length of any cycle.
        k (int): Each node is connected to k nearest neighbors in the ring. Default is 2.
        p (float): Probability of adding a shortcut edge. Default is 0.5.

    Returns:
        np.ndarray: Adjacency matrix of the directed cyclic graph.
    """
    while True:
        def degree(var):
            """Calculate the degree of a node in the adjacency matrix."""
            return np.sum(B[:, var]) + np.sum(B[var, :])

        def reachable(end, start):
            """Check if 'end' is reachable from 'start' using DFS."""
            seen = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node == end:
                    return True
                if node not in seen:
                    seen.add(node)
                    neighbors = np.where(B[node, :])[0]
                    stack.extend(neighbors)
            return False

        def find_scc(var):
            """Find variables in the strongly connected component (SCC) of 'var'."""
            scc = {var}
            for i in range(n_vars):
                if i != var and reachable(var, i) and reachable(i, var):
                    scc.add(i)
            return scc

        # Generate undirected Newman-Watts-Strogatz graph
        G = nx.newman_watts_strogatz_graph(n_vars, k, p)

        # Initialize adjacency matrix
        B = np.zeros((n_vars, n_vars), dtype=int)

        # Randomly assign directions to the edges
        for u, v in G.edges():
            if np.random.rand() < 0.5:
                B[u, v] = 1
            else:
                B[v, u] = 1

        # Add additional edges while respecting constraints
        for i in range(n_vars):
            for j in np.random.permutation(n_vars):
                if i == j or B[i, j] or B[j, i]:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                # Randomly assign direction
                if np.random.choice([True, False]):
                    B[i, j] = 1
                else:
                    B[j, i] = 1
                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    if B[i, j]:
                        B[i, j] = 0
                    else:
                        B[j, i] = 0

        # Validate degrees
        if any(degree(i) > max_degree for i in range(n_vars)):
            continue

        # Check cyclicity
        G = nx.DiGraph(B)
        if nx.is_directed_acyclic_graph(G):  # Ensure the graph contains cycles
            continue

        return B


####################### data generation functions #######################

def sample_W(B, B_low=0.6, B_high=1.0, var_low=0.75, var_high=1.25,
                      flip_sign=True, max_eig=1.0, max_cond_number=100):
    """
    Generate graph parameters given the B matrix by sampling uniformly in a range.

    Edge weights are sampled independently from Unif[-2, -1.0] ∪ [1.0, 2.0].

    Args:
        B (np.ndarray): B binary adjacency matrix of the graph.
        B_low (float): Lower bound for edge weights.
        B_high (float): Upper bound for edge weights.
        var_low (float): Lower bound for scale/sigma of the noise for each variable.
        var_high (float): Upper bound for scale/sigma of the noise for each variable.
        flip_sign (bool): Whether to randomly flip the sign of edge weights.
        max_eig (float): Maximum allowed absolute eigenvalue for stability.
        max_cond_number (float): Maximum allowed condition number for invertibility.

    Returns:
        tuple: Tuple containing:
            - W (np.ndarray): Weighted adjacency matrix with sampled edge weights.
            - sigmas (np.ndarray): Scales/Sigmas of the noise for each variable.
    """
    assert B.shape[0] == B.shape[1]
    n_variables = B.shape[0]

    stable = False
    while not stable:
        W = B * np.random.uniform(B_low, B_high, size=B.shape)
        if flip_sign:
            signs = np.random.choice([-1, 1], size=W.shape)
            W *= signs
        eigenvalues = np.linalg.eigvals(W)
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        cond_number = np.linalg.cond(np.eye(n_variables) - W)
        stable = max_abs_eigenvalue < max_eig and cond_number < max_cond_number

    sigmas = np.random.uniform(var_low, var_high, size=n_variables)
    #return W.T, sigmas
    return W, sigmas

def sample_data(W, num_samples=5000, noise_type='gauss-ev', scale=None, scales=None):
    """
    Generate data given the weighted adjacency matrix and noise type.

    Args:
        W (np.ndarray): Weighted adjacency matrix.
        num_samples (int): Number of samples to generate.
        noise_type (str): Type of noise to use ('gauss-ev', 'exp', 'softplus').
        scale (float): Scale of the noise, essentially the standard deviation/sigma.
        scales (np.ndarray): Scales of the noise for each variable.

    Returns:
        np.ndarray: Data matrix of shape (n_samples, n_variables).
        np.ndarray: Precision matrix if Gaussian noise; otherwise None.
    """

    # raise an error if noise_type is 'gauss-ev' and not all scales are the same
    if noise_type == 'gauss-ev' and scales is not None and not np.allclose(scales, scales[0]):
        raise ValueError("Gaussian noise requires equal scales for all variables.")

    d = W.shape[0]

    # if neither scale nor scales are provided, use scales all 1.0
    if scales is None and scale is None:
        scales = np.ones(d)
    elif scales is None:
        scales = np.ones(d) * scale
    elif scale is not None:
        raise ValueError("Provide either 'scale' or 'scales', not both.")

    # Construct SIGMA
    SIGMA = np.diag(scales)  # Variance or scale matrix (diagonal)

    # Generate standardized noise
    if noise_type == 'gauss-ev':
        std_noise = np.random.randn(num_samples, d)  # Gaussian noise with mean 0, std 1
    elif noise_type == 'exp':
        #std_noise = np.random.exponential(scale=1.0, size=(num_samples, d)) - 1  # Centered exponential noise with mean 0
        std_noise = np.random.exponential(scale=1.0, size=(num_samples, d))
    elif noise_type == 'softplus':
        std_noise = np.random.normal(scale=scales, size=(num_samples, d))
        std_noise = np.log(1 + np.exp(std_noise))
    else:
        raise ValueError(f"Invalid noise type: {noise_type}. Must be 'gauss-ev', 'exp', or 'softplus'.")

    # Apply scaling by SIGMA**0.5
    scaled_noise = std_noise @ np.sqrt(SIGMA)

    # Solve for X using np.linalg.solve for numerical stability
    if np.linalg.cond(np.eye(d) - W.T) > 1e10:
        raise ValueError("Matrix (I - W.T) is nearly singular. Adjust W.")
    X = np.linalg.solve(np.eye(d) - W.T, scaled_noise.T).T

    # requires transposing W to get the correct X
    # # Solve for X using np.linalg.solve for numerical stability
    # if np.linalg.cond(np.eye(d) - W) > 1e10:
    #     raise ValueError("Matrix (I - W) is nearly singular. Adjust W.")
    # X = np.linalg.solve(np.eye(d) - W, scaled_noise.T).T

    # Compute precision matrix for Gaussian noise
    if noise_type == 'gauss-ev':
        print("Generating precision matrix for Gaussian noise.")
        #prec_matrix = (np.eye(d) - W.T) @ np.linalg.inv(SIGMA) @ (np.eye(d) - W.T).T
        prec_matrix = (np.eye(d) - W.T) @ np.diag(1 / np.diag(SIGMA)) @ (np.eye(d) - W.T).T # computationally more efficient than using np.linalg.inv
        
    else:
        prec_matrix = None

    return X, prec_matrix

def _sample_data(W, sigmas=None, num_samples=5000, softplus=False):
    """
    Generate data given the B matrix and noise scales.

    Args:
        W (np.ndarray): Weighted adjacency matrix.
        sigmas (np.ndarray): These are standard deviations of the noise for each variable.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Data matrix of shape (n_samp, n_variables).
    """
    # number of variables
    d = W.shape[0]

    # check if softplus is True, we overwrite potentially given sigmas
    if softplus:
        sigmas = np.ones(W.shape[0]) # here we use standard deviation of 1 for each variable because we will apply softplus transformation to vanilla Gaussian noise

    n_var = len(sigmas)
    # TODO: change this to sampling Gaussian EV noise because it is an identifiable SEM by causal identification theorems (see Peters et al. 2014)

    # option A: generate noise from normal(0, sigmas) (Gaussian noise with equal variance for all variables meaning sigmas = [sigma, sigma, ..., sigma])
    if not softplus:
        #noise = (np.random.uniform(-np.array(sigmas)[:, None], np.array(sigmas)[:, None], size=(n_var, num_samples)).T)**5 # here we reshape 1D sigmas vector (n_var,) to 2D sigmas matrix (n_var, 1)
        #noise = np.random.normal(scale=sigmas, size=(num_samples, n_var))
        noise = np.random.normal(scale=sigmas, size=(num_samples, n_var))

    # option B: generate noise from softplus(normal(0, sigmas))
    else:        
        noise = np.random.normal(scale=sigmas, size=(num_samples, n_var))
        noise = np.log(1 + np.exp(noise))
    
    # compute X = noise @ (I - W.T)^(-1)
    # original using W.T
    #X = np.linalg.solve(np.eye(n_var) - W.T, noise.T).T # solves linear system A @ X.T = noise.T s.t. X.T = A^(-1) @ noise.T, where A = I - W.T. We traonspose result to get X instead of X.T
    # modified using W
    X = np.linalg.solve(np.eye(d) - W, noise.T).T # solves linear system A @ X = noise s.t. X = A^(-1) @ noise, where A = I - W

    # compute the ground truth precision matrix using the true W matrix and sigmas
    prec_matrix = (np.eye(d) - W) @ np.diag(1 / sigmas) @ (np.eye(d) - W).T

    return X, prec_matrix