import numpy as np
import networkx as nx
import igraph as ig
import random
from scipy.special import expit as sigmoid

# Import utility functions
from causal_helpers import set_random_seed

########################## DAG Sampling Functions ##########################

def sample_B(n_nodes, n_edges, graph_type, ws_nei=None, ws_p=None):
    """Simulate a random DAG with a specified number of nodes and edges.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    n_edges : int
        Expected number of edges.
    graph_type : str
        Type of graph: "ER", "SF", "BP", or "WS".
    ws_nei : int, optional
        Number of neighbors for WS graph (default: max(2, n_nodes // 10)).
    ws_p : float, optional
        Rewiring probability for WS graph (default: 0.1).

    Returns
    -------
    B : array-like, shape (n_nodes, n_nodes)
        Binary adjacency matrix of the DAG.
    """

    def _random_permutation(M):
        """Apply a random permutation to adjacency matrix M."""
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        """Convert undirected adjacency matrix to DAG by enforcing acyclicity."""
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        """Convert an igraph graph to an adjacency matrix."""
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        G_und = ig.Graph.Erdos_Renyi(n=n_nodes, m=n_edges) # can have n_nodes*(n_nodes-1)/2 edges at most
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

    elif graph_type == "SF":
        G = ig.Graph.Barabasi(n=n_nodes, m=int(round(n_edges / n_nodes)), directed=True)
        B = _graph_to_adjmat(G)

    elif graph_type == "BP":
        top = int(0.2 * n_nodes)
        G = ig.Graph.Random_Bipartite(top, n_nodes - top, m=n_edges, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)

    elif graph_type == "WS":
        # Set default values for Watts-Strogatz parameters
        if ws_nei is None:
            ws_nei = max(2, n_nodes // 10)  # At least 2 neighbors, or 10% of nodes
        if ws_p is None:
            ws_p = 0.1  # Default rewiring probability

        G_und = ig.Graph.Watts_Strogatz(dim=1, size=n_nodes, nei=ws_nei, p=ws_p)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)  # Ensure acyclicity

    else:
        raise ValueError("Unknown graph type. Choose from ['ER', 'SF', 'BP', 'WS'].")

    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag(), "Generated graph is not a DAG!"
    return B_perm


def is_DAG(B):
    """Check if B is a dag or not.

    Parameters
    ----------
    B : array-like, shape (n_nodes, n_nodes)
        Binary adjacency matrix of DAG, where ``n_nodes``
        is the number of features.

    Returns
    -------
     G: boolean
        Returns true or false.

    """
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    return G.is_dag()

def is_dag(B):
    """Check if B is a dag or not.

    Parameters
    ----------
    B : array-like, shape (n_nodes, n_nodes)
        Binary adjacency matrix of DAG, where ``n_nodes``
        is the number of features.

    Returns
    -------
     G: boolean
        Returns true or false.

    """
    G = ig.Graph.Weighted_Adjacency(B.tolist())
    return G.is_dag()


def sample_W(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : array-like, shape (d, d)
        Binary adjacency matrix of DAG, where ``d``
        is the number of features.
    w_ranges : tuple
        Disjoint weight ranges.

    Returns
    -------
    W : array-like, shape (d, d)
        Weighted adj matrix of DAG, where ``d``
        is the number of features.
    """

    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_mixed_sem(
    W, n_samples, is_cont, noise_scale=None, sem_type='mixed_random_i_dis'
):
    """Simulate mixed samples from linear SEM with specified type of noise.

    Parameters
    ----------
    W : array-like, shape (n_nodes, n_nodes)
        Weighted adjacency matrix of DAG, where ``n_nodes``
        is the number of variables.
    n_samples : int
        Number of samples. n_samples=inf mimics population risk.
    is_cont : array-like, shape (1, n_nodes)
        Indicator of discrete/continuous variables, where "1"
        indicates a continuous variable, while "0" a discrete
        variable.
    noise_scale : float, optional
        Scale parameter of additive noise (default: None).
    sem_type : str, optional
        SEM type. Options are 'gauss' or 'mixed_random_i_dis' (default: 'mixed_random_i_dis').

    Returns
    -------
    X : array-like, shape (n_samples, n_nodes)
        Data generated from linear SEM with specified type of noise,
        where ``n_nodes`` is the number of variables.
    """

    def _simulate_single_equation(X, w, scale, is_cont_j):
        """Simulate samples from a single equation.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_nodes_parents)
            Data of parents for a specified variable, where
            n_nodes_parents is the number of parents.
        w : array-like, shape (1, n_nodes_parents)
            Weights of parents.
        scale : scale parameter of additive noise.
        is_cont_j : indicator of the j^th variable. If 1, the variable is continuous; if 0, the variable is discrete.
        Returns
        -------
        x : array-like, shape (n_samples, 1)
                    Data for the specified variable.
        """
        if sem_type == "gauss": # fully continuous SEM
            z = np.random.normal(scale=scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "mixed_random_i_dis": # mixed SEM with some continuous and some discrete variables
            # randomly generated with fixed number of discrete variables.
            if is_cont_j:  # 1:continuous;   0:discrete
                # option A (Laplace)
                #z = np.random.laplace(0, scale=scale, size=n_samples) # Laplace noise (non-Gaussian assumption for lingam based methods)
                # option B (softplus(Normal))
                z = np.random.normal(scale=scale, size=n_samples)   # Softplus(Normal) noise as done in csuite benchmark, also non-Gaussian
                z = np.log(1 + np.exp(z))
                x = X @ w + z
            else:
                prob = (np.tanh(X @ w) + 1) / 2
                x = np.random.binomial(1, p=prob) * 1.0 # whenever a variable j has no parents, then it is generated from a Bernoulli distribution with probability sigmoid(0)
        else:
            raise ValueError("unknown sem type")
        return x
    
    n_nodes = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(n_nodes)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(n_nodes) # equal variances for all variables
    else:
        if len(noise_scale) != n_nodes:
            raise ValueError("noise scale must be a scalar or has length n_nodes")
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError("W must be a DAG")
    if np.isinf(n_samples):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/n_nodes X'X = true cov
            X = (
                np.sqrt(n_nodes)
                * np.diag(scale_vec)
                @ np.linalg.inv(np.eye(n_nodes) - W)
            )
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_nodes
    X = np.zeros([n_samples, n_nodes])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        # X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        X[:, j] = _simulate_single_equation(
            X[:, parents], W[parents, j], scale_vec[j], is_cont[0, j]
        )
    return X