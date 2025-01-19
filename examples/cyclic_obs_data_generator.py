# sampling observational data from cyclic graphs

import numpy as np

def sample_er_dcg(n_vars, max_degree, max_cycle):
    """
    Randomly sample a Directed Cyclic Graph (DCG) with a maximum degree and cycle length.

    Args:
        n_vars (int): Number of variables (nodes) in the graph.
        max_degree (int): Maximum degree (number of edges) per node.
        max_cycle (int): Maximum allowed length of any cycle in the graph.

    Returns:
        np.ndarray: Adjacency matrix representing the graph structure.
    """
    support = np.zeros((n_vars, n_vars))

    def degree(var):
        return len(np.where(support[:, var])[0]) + len(np.where(support[var, :])[0])

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
                neighbors = np.where(support[node, :])[0]
                stack.extend(neighbors)
        return False

    def find_scc(var):
        """Find variables in the strongly connected component (SCC) of 'var'."""
        scc = set([var])
        for i in range(support.shape[0]):
            if i != var and reachable(var, i) and reachable(i, var):
                scc.add(i)
        return scc

    # Assign maximum degrees randomly
    max_degrees = np.random.randint(1, max_degree + 1, size=n_vars)

    for i in range(n_vars):
        for j in np.random.permutation(n_vars):
            if i == j:
                continue
            if degree(i) >= max_degrees[i]:
                break
            direction = np.random.choice([True, False])
            if degree(j) < max_degrees[j]:
                if direction:
                    support[i, j] = 1
                else:
                    support[j, i] = 1
                # Check cycle length constraint
                if len(find_scc(j)) > max_cycle:
                    if direction:
                        support[i, j] = 0
                    else:
                        support[j, i] = 0

    # Validate degrees
    for i in range(n_vars):
        assert degree(i) <= max_degree, f"Node {i} exceeds max degree."

    return support

def sample_W(B, B_low=1.0, B_high=2.0, var_low=0.5, var_high=1.5,
                      flip_sign=True, max_eig=1.0, max_cond_number=100):
    """
    Generate graph parameters given the support matrix by sampling uniformly in a range.

    Edge weights are sampled independently from Unif[-2, -1.0] ∪ [1.0, 2.0].

    Args:
        B (np.ndarray): Support (adjacency) matrix of the graph.
        B_low (float): Lower bound for edge weights.
        B_high (float): Upper bound for edge weights.
        var_low (float): Lower bound for error variances.
        var_high (float): Upper bound for error variances.
        flip_sign (bool): Whether to randomly flip the sign of edge weights.
        max_eig (float): Maximum allowed absolute eigenvalue for stability.
        max_cond_number (float): Maximum allowed condition number for invertibility.

    Returns:
        tuple: Tuple containing:
            - W (np.ndarray): Weighted adjacency matrix with sampled edge weights.
            - noise_scales (np.ndarray): Noise variances for each variable.
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

    noise_scales = np.random.uniform(var_low, var_high, size=n_variables)
    return W, noise_scales

def sample_data(B, noise_scales, num_samples):
    """
    Generate data given the B matrix and noise scales.

    Args:
        B (np.ndarray): Weighted adjacency matrix.
        noise_scales (np.ndarray): Noise variances for each variable.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Data matrix of shape (n_samp, n_variables).
    """
    n_var = len(noise_scales)
    noise = (np.random.uniform(-np.array(noise_scales)[:, None], np.array(noise_scales)[:, None], size=(n_var, num_samples)).T)**5 # here we reshape 1D noise_scales vector (n_var,) to 2D noise_scales matrix (n_var, 1)
    X = np.linalg.solve(np.eye(n_var) - B.T, noise.T).T # solves AX = B, where A = I - B.T and B = N.T, numerically stable
    #X = (np.linalg.inv(np.eye(n_variables) - B.T) @ N.T).T # inverts A = I - B.T and multiplies by B = N.T
    return X
