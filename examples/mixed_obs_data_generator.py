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
        G_und = ig.Graph.Erdos_Renyi(n=n_nodes, m=n_edges, directed=True)
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
