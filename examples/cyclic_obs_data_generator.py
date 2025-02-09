# sampling observational data from cyclic graphs

import numpy as np
import networkx as nx


########################## graph sampling functions ##########################

# ER cycle graph sampling
def sample_ER_dcg(d, max_degree, max_cycle, n_edges=None, p=0.2, edge_add_prob=0.3):
    """
    Sample a Directed Cyclic Graph (DCG) with an Erdős-Rényi degree distribution.

    Args:
        d (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (number of edges) per node.
        max_cycle (int): Maximum allowed length of any cycle in the graph.
        p (float): Probability of edge creation in the Erdős-Rényi graph.
        n_edges (int): Number of edges in the graph. If provided, take precedence over p.
        edge_add_prob (float): Probability of adding an extra cyclic edge.

    Returns:
        np.ndarray: Adjacency matrix representing the graph structure.
    """
    #while True:  # Repeat until a valid cyclic graph is generated
    n_tries = 10
    for _ in range(n_tries):  # Repeat until a valid cyclic graph is generated
        def degree(var):
            """Calculate the degree of a node in the adjacency matrix."""
            return np.count_nonzero(B[:, var]) + np.count_nonzero(B[var, :])

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
        if n_edges is not None:
            _G = nx.gnm_random_graph(d, n_edges, directed=True)  # Use number of edges parameter
        elif n_edges is None and p is not None:
            _G = nx.erdos_renyi_graph(d, p=p, directed=True)  # Use density parameter
        else:
            raise ValueError("Provide either 'n_edges' or 'p'.")

        # Remove cycles by ensuring edges are only added in topological order
        G_edges = [(u, v) for u, v in _G.edges() if u < v]
        G = nx.DiGraph(G_edges)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Add any missing nodes to ensure the graph has exactly d nodes
        for node in range(d):
            if node not in G.nodes:
                G.add_node(node)

        # Convert to adjacency matrix
        B = nx.to_numpy_array(G)

        # 🔥 Add cyclic edges while limiting density
        for i in np.random.permutation(d):  # Random order
            for j in np.random.permutation(d):
                if i == j or B[i, j] == 1 or B[j, i] == 1:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                if np.random.rand() > edge_add_prob:  # 🔥 Skip edge with probability
                    continue
                direction = np.random.choice([True, False])
                B[i, j] = direction
                B[j, i] = 1 - direction  # Ensure directed nature

                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    B[i, j], B[j, i] = 0, 0  # Revert edge if cycle is too large

        # Validate degrees
        for i in range(d):
            assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

        # Check if the graph is cyclic using NetworkX
        G = nx.DiGraph(B)
        is_cyclic = not nx.is_directed_acyclic_graph(G)  # True if cyclic
        if is_cyclic:
            return B
        
    raise ValueError(
        f"Failed to generate a cyclic ER graph after {n_tries} attempts. "
        f"Parameters: d={d}, max_degree={max_degree}, max_cycle={max_cycle}, "
        f"n_edges={n_edges}, p={p}, edge_add_prob={edge_add_prob}."
    )


# SF cycle graph sampling
def sample_SF_dcg(d, max_degree, max_cycle, n_edges=None, p=0.2, edge_add_prob=0.3):
    """
    Sample a Directed Cyclic Graph (DCG) with a scale-free degree distribution.

    Args:
        d (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (number of edges) per node.
        max_cycle (int): Maximum allowed length of any cycle in the graph.
        n_edges (int): Number of edges in the graph. If provided, take precedence over p.
        p (float): Probability value for beta in the scale-free graph such that alpha+beta+gamma=1.
        edge_add_prob (float): Probability of adding an extra cyclic edge.

    Returns:
        np.ndarray: Adjacency matrix representing the graph structure.
    """
    n_tries = 10
    for _ in range(n_tries):  # Repeat until a valid cyclic graph is generated
    #while True:  # Repeat until a valid cyclic graph is generated
        def degree(var):
            """Calculate the degree of a node in the adjacency matrix."""
            return np.count_nonzero(B[:, var]) + np.count_nonzero(B[var, :])

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

        # Generate a scale-free graph
        if n_edges is not None:
            m = n_edges // d  # Average degree
            _G = nx.barabasi_albert_graph(d, m)  # Undirected scale-free graph
        elif n_edges is None and p is not None:
            beta = p
            gamma = 0.05
            alpha = 1 - beta - gamma
            _G = nx.scale_free_graph(d, alpha=alpha, beta=beta, gamma=gamma)  # Directed scale-free graph
        else:
            raise ValueError("Provide either 'n_edges' or 'p (beta value)'.")

        # Convert to directed graph while preserving scale-free properties
        G_edges = [(u, v) for u, v in _G.edges() if u < v]
        G = nx.DiGraph(G_edges)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Ensure exactly `d` nodes exist
        for node in range(d):
            if node not in G.nodes:
                G.add_node(node)

        # Convert to adjacency matrix
        B = nx.to_numpy_array(G)

        # 🔥 Add cyclic edges while controlling density
        for i in np.random.permutation(d):  # Randomized node order
            for j in np.random.permutation(d):
                if i == j or B[i, j] == 1 or B[j, i] == 1:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                if np.random.rand() > edge_add_prob:  # 🔥 Skip edge with probability
                    continue
                direction = np.random.choice([True, False])
                B[i, j] = direction
                B[j, i] = 1 - direction  # Ensure directed nature

                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    B[i, j], B[j, i] = 0, 0  # Revert edge if cycle is too large

        # Validate degrees
        for i in range(d):
            assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

        # Check if the graph is cyclic
        G = nx.DiGraph(B)
        is_cyclic = not nx.is_directed_acyclic_graph(G)  # True if cyclic
        if is_cyclic:
            return B

    raise ValueError(
        f"Failed to generate a cyclic SF graph after {n_tries} attempts. "
        f"Parameters: d={d}, max_degree={max_degree}, max_cycle={max_cycle}, "
        f"n_edges={n_edges}, p={p}, edge_add_prob={edge_add_prob}."
    )


# NWS cycle graph sampling
def sample_NWS_dcg(d, max_degree, max_cycle, k=2, p=0.2, edge_add_prob=0.3):
    """
    Sample a Directed Cyclic Graph (DCG) using the Newman-Watts-Strogatz model.

    Args:
        d (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (in + out) per node.
        max_cycle (int): Maximum allowed length of any cycle.
        k (int): Each node is connected to k nearest neighbors in the ring. Default is 2.
        p (float): Probability of adding a shortcut edge. Default is 0.2.
        edge_add_prob (float): Probability of adding an extra cyclic edge.

    Returns:
        np.ndarray: Adjacency matrix of the directed cyclic graph.
    """
    n_tries = 10
    for _ in range(n_tries):
    #while True:
        def degree(var):
            """Calculate the degree of a node in the adjacency matrix."""
            return np.count_nonzero(B[:, var]) + np.count_nonzero(B[var, :])

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
            for i in range(d):
                if i != var and reachable(var, i) and reachable(i, var):
                    scc.add(i)
            return scc

        # Generate an **undirected** Newman-Watts-Strogatz graph
        G = nx.newman_watts_strogatz_graph(d, k, p)  # Undirected base graph

        # Initialize adjacency matrix
        B = np.zeros((d, d), dtype=int)

        # 🔥 Convert to **directed** graph by randomly assigning edge directions
        for u, v in G.edges():
            if np.random.rand() < 0.5:
                B[u, v] = 1
            else:
                B[v, u] = 1

        # 🔥 Add **cyclic** edges while respecting constraints
        for i in np.random.permutation(d):  # Randomized node order
            for j in np.random.permutation(d):
                if i == j or B[i, j] == 1 or B[j, i] == 1:  # No self-loops or duplicate edges
                    continue
                if degree(i) >= max_degree:
                    break
                if degree(j) >= max_degree:
                    continue
                if np.random.rand() > edge_add_prob:  # 🔥 Skip edge with probability
                    continue
                direction = np.random.choice([True, False])
                B[i, j] = direction
                B[j, i] = 1 - direction  # Ensure directed nature

                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    B[i, j], B[j, i] = 0, 0  # Revert edge if cycle is too long

        # Validate degrees
        for i in range(d):
            assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

        # Check if the graph is cyclic using NetworkX
        G = nx.DiGraph(B)
        is_cyclic = not nx.is_directed_acyclic_graph(G)  # True if cyclic
        if is_cyclic:
            return B

    raise ValueError(
        f"Failed to generate a cyclic NWS graph after {n_tries} attempts. "
        f"Parameters: d={d}, max_degree={max_degree}, max_cycle={max_cycle}, "
        f"k_nws={k}, p={p}, edge_add_prob={edge_add_prob}."
    )


####################### data generation functions #######################

def sample_W(B, B_low=0.6, B_high=1.0, var_low=0.75, var_high=1.25,
                      flip_sign=True, max_eig=1.0, max_cond_number=100):
    """
    Generate graph parameters given the B matrix by sampling uniformly in a range.

    Edge weights are sampled independently from Unif[-1.0, -0.6] or Unif[0.6, 1.0].

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
    return W, sigmas

def sample_data(W, num_samples=5000, noise_type='GAUSS-EV', scale=None, scales=None):
    """
    Generate data given the weighted adjacency matrix and noise type.

    Args:
        W (np.ndarray): Weighted adjacency matrix of shape (d, d).
        num_samples (int): Number of samples to generate.
        noise_type (str): Type of noise to use ('GAUSS-EV', 'EXP', 'SOFTPLUS', 'UNIFORM').
        scale (float): Baseline noise scale.
        scales (np.ndarray): Per-variable noise scales (diagonal of SIGMA).

    Returns:
        np.ndarray: Data matrix of shape (n_samples, n_variables).
        np.ndarray: Precision matrix for Gaussian noise, otherwise None.
    """
    d = W.shape[0]

    # Set default scales
    if scales is None and scale is None:
        scales = np.ones(d)
    elif scales is None:
        scales = np.ones(d) * scale
    elif scale is not None:
        raise ValueError("Provide either 'scale' or 'scales', not both.")
    if np.any(scales <= 0):
        raise ValueError("All scales must be positive.")

    # Construct SIGMA
    SIGMA = np.diag(scales)

    # Generate standardized noise
    if noise_type == 'GAUSS-EV':
        std_noise = np.random.randn(num_samples, d)
    elif noise_type == 'EXP':
        std_noise = np.random.exponential(scale=1.0, size=(num_samples, d)) - 1
    elif noise_type == 'SOFTPLUS':
        std_noise = np.random.normal(scale=scales, size=(num_samples, d))
        std_noise = np.log(1 + np.exp(std_noise))  # Apply softplus

        # Correct the mean to center the noise around zero
        expected_mean = np.log(2) + (scales**2 / 2)  # E[softplus(X)]
        std_noise -= expected_mean  # Centering transformation

        # Compute variance of softplus-transformed noise
        softplus_variance = (np.pi**2 / 24) + (scales**4 / 4)
        SIGMA = np.diag(softplus_variance)  # Adjust SIGMA for correct scaling

    elif noise_type == 'UNIFORM':
        std_noise = np.random.uniform(-1, 1, size=(num_samples, d))
    else:
        raise ValueError(f"Invalid noise type: {noise_type}. Must be 'GAUSS-EV', 'EXP', 'SOFTPLUS', or 'UNIFORM'.")

    # Apply scaling by SIGMA
    scaled_noise = std_noise @ np.sqrt(SIGMA)

    # Solve for X using np.linalg.solve for numerical stability
    if np.linalg.cond(np.eye(d) - W.T) > 1e10:
        raise ValueError("Matrix (I - W.T) is nearly singular. Adjust W.")
    X = np.linalg.solve(np.eye(d) - W.T, scaled_noise.T).T
    # Using W.T enables interpretation of W_ji as effect of X_j on X_i (row index = parent, column index = child)
    # Using W would imply W_ji as effect of X_i on X_j (row index = child, column index = parent)
    # X and scale_noise are nxd matrices, np.eye(d) - W.T is a dxd matrix, the results will have same shape as second argument of np.linalg.solve
    # which is scaled_noise.T in this case, thus we need to transpose the result to get the nxd matrix X

    # Compute precision matrix
    prec_matrix = None
    Q = np.eye(d) - W.T
    if noise_type == 'GAUSS-EV':
        # option A:
        # noise_cov = np.diag(scales**2)
        # prec_matrix = Q @ np.linalg.inv(noise_cov) @ Q.T
        # option B:
        prec_matrix = Q @ np.diag(1 / scales**2) @ Q.T
    elif noise_type == 'EXP':
        noise_cov = np.diag(scales**2)  # Approximation for exponential noise
        prec_matrix = Q @ np.linalg.inv(noise_cov) @ Q.T
    elif noise_type == 'SOFTPLUS':
        noise_cov = np.diag(softplus_variance)  # Use the analytical variance
        prec_matrix = Q @ np.linalg.inv(noise_cov) @ Q.T
    elif noise_type == 'UNIFORM':
        noise_cov = np.diag(scales**2 / 3)  # Variance of U(-a, a) is a^2 / 3
        prec_matrix = Q @ np.linalg.inv(noise_cov) @ Q.T

    return X, prec_matrix
