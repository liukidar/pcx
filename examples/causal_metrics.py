import lingam
import networkx as nx
import numpy as np
from gadjid import ancestor_aid, parent_aid, shd
from cdt.metrics import SHD
from dglearn.dglearn.evaluation.evaluation_shd import min_colperm_shd as _compute_cycle_SHD
from dglearn.dglearn.evaluation.evaluation_kld import minimize_kld as _minimize_kld
from scipy.optimize import minimize, Bounds, basinhopping

########################################################################################################

def compute_SHD_robust(B_true, B_est):
    """
    Compute Structural Hamming Distance (SHD) and normalised SHD.
    Despite the graph being acyclic or cyclic, this function computes the SHD robustly.

    Parameters:
        B_true (np.ndarray): Ground truth adjacency matrix.
        B_est (np.ndarray): Estimated adjacency matrix.

    Returns:
        tuple: (normalised SHD, absolute SHD)
    """
    B_true_binary = (B_true > 0).astype(np.int8)
    B_est_binary = (B_est > 0).astype(np.int8)
    # number of edges in the ground truth graph 
    num_edges = np.sum(B_true_binary)
    absolute_SHD = SHD(B_true_binary, B_est_binary, double_for_anticausal=False)
    normalised_SHD = absolute_SHD / num_edges
    return normalised_SHD, absolute_SHD


def error_metrics(B_true, B_est):
    # Computes the true positive, true negative, false positive and false negative counts between the true and estimated adjacency matrices.
    # Parameters:
    # 1) B_true - Ground truth adjacency matrix.
    # 2) B_est - Estimated adjacency matrix.
    # Both the matrices are assumed to binary. 

    B_true = B_true * 1.0
    B_est = B_est * 1.0

    acc = (B_true == B_est).sum()
    tp_matrix = B_true * B_est
    fp_matrix = (B_est - tp_matrix).sum()
    tp, tn, fp, fn = tp_matrix.sum(), acc - tp_matrix.sum(), fp_matrix.sum(), (B_true - tp_matrix).sum() 
    
    return tp, tn, fp, fn

def compute_F1_skeleton(B_true, B_est):
    """
    Calculate F1 score for the skeleton (undirected graph).

    Args:
        B_true (np.ndarray): Ground truth adjacency matrix (d x d).    
        B_est (np.ndarray): Predicted adjacency matrix (d x d).

    Returns:
        precision (float): Precision for the skeleton.
        recall (float): Recall for the skeleton.
        F1 (float): F1 score for the skeleton.
    """
    # Ensure no self-loops
    np.fill_diagonal(B_est, 0)
    np.fill_diagonal(B_true, 0)

    # Convert to undirected skeleton
    B_est_skeleton = (B_est + B_est.T) > 0
    B_true_skeleton = (B_true + B_true.T) > 0

    # Flatten to 1D arrays
    y_pred = B_est_skeleton.flatten()
    y_true = B_true_skeleton.flatten()

    # Compute TP, FP, FN
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # Precision, Recall, F1
    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return precision, recall, f1

def compute_F1_directed(B_true, B_est):
    """ 
    Compute the F1 score between the true and estimated binary adjacency matrices respecting the directionality of the edges.

    Args:
    - B_true (np.ndarray): The true binary adjacency matrix.
    - B_est (np.ndarray): The estimated binary adjacency matrix.

    Returns:
    - f1 (float): The F1 score between the true and estimated binary adjacency matrices
    """
    tp, tn, fp, fn = error_metrics(B_true, B_est)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_AUROC(B_true, W_est, n_points=50):
    # Computes the Area Under Precision Recall Curve (AUPR)
    # B_true - binary matrix (ground truth)
    # W_est - Positive entry matrix (estimated) ie weighted adjacency matrix
    # n_points - (int) - number of points to compute precision and recall

    max_threshold = W_est.max()
    threshold_list = np.linspace(0, max_threshold, n_points)
    
    tpr_list, fpr_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = error_metrics(B_true, W_est >= threshold)
        tp_rate = tp / (tp + fn)
        fp_rate = fp / (tn + fp)
        tpr_list.append(tp_rate)
        fpr_list.append(fp_rate)
    
    area = np.trapz(tpr_list[::-1], fpr_list[::-1])
    return tpr_list, fpr_list, area

def compute_AUPRC(B_true, W_est, n_points=50):

    max_threshold = W_est.max()
    threshold_list = np.linspace(0, max_threshold, n_points)
    rec_list, pre_list = list(), list()
    for threshold in threshold_list:
        tp, tn, fp, fn = error_metrics(B_true, W_est >= threshold)
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        rec_list.append(rec)
        pre_list.append(pre)

    rec_list.append(0)
    pre_list.append(1.0)

    area = np.trapz(pre_list[::-1], rec_list[::-1])
    baseline = B_true.sum() / (B_true.shape[0] * B_true.shape[1])

    return baseline, area

# Function to compute BIC and AIC
def compute_model_fit(W, X):
    """
    Compute BIC and AIC scores for a given adjacency matrix and observational data.

    Parameters:
        W (np.ndarray): Weighted adjacency matrix. Important that W is weighted as its used to define SEM description (see 'desc' in lingam) for semopy.
        X (pd.DataFrame): Observational data as a Pandas DataFrame.

    Returns:
        tuple: BIC and AIC scores as floats.
    """
    # Define is_ordinal based on the number of unique values in each column of X
    is_ordinal = np.array([0 if len(np.unique(X[:, col])) > 2 else 1 for col in range(X.shape[1])]).tolist() # 0 for continuous, 1 for ordinal


    # Evaluate model fit using LiNGAM's utility
    fit_scores = lingam.utils.evaluate_model_fit(W, X, is_ordinal=is_ordinal)
    bic = fit_scores['BIC'].values[0]
    aic = fit_scores['AIC'].values[0]
    return bic, aic

#################################### CYCLIC GRAPH METRICS ######################################

def compute_cycle_F1(B_true, B_est, max_cycle_length=3, verbose=False):
    """
    Compute the F1 score based on all edges involved in cycles in B_true and B_est.

    Parameters:
        B_true (np.ndarray): The true directed cyclic graph.
        B_est (np.ndarray): The estimated directed cyclic graph.
        max_cycle_length (int): The maximum length of cycles to consider.
        verbose (bool): If True, print debug information.

    Returns:
        float: The F1 score based on cycle-related edges.
    """
    # Convert input arrays to directed graphs
    G_true = nx.DiGraph(B_true)
    G_est = nx.DiGraph(B_est)

    # Ensure B_true contains cycles
    true_cycles = list(nx.simple_cycles(G_true, length_bound=max_cycle_length))
    if not true_cycles:
        raise ValueError("B_true is not cyclic. Please provide a cyclic true graph.")

    # Ensure B_est contains cycles
    est_cycles = list(nx.simple_cycles(G_est, length_bound=max_cycle_length))
    if not est_cycles:
        print("B_est is not cyclic. Returning 0 F1-score.")
        return 0.0

    if verbose:
        print(f"True cycles: {true_cycles}")
        print(f"Estimated cycles: {est_cycles}")

    # Merge all edges from cycles in B_true into a single set
    true_cycle_edges = set()
    for cycle in true_cycles:
        for i in range(len(cycle)):
            true_cycle_edges.add(frozenset((cycle[i], cycle[(i + 1) % len(cycle)])))

    # Merge all edges from cycles in B_est into a single set
    est_cycle_edges = set()
    for cycle in est_cycles:
        for i in range(len(cycle)):
            est_cycle_edges.add(frozenset((cycle[i], cycle[(i + 1) % len(cycle)])))

    # Compute true positives (matching edges), false positives, and false negatives
    tp = len(true_cycle_edges & est_cycle_edges)
    fp = len(est_cycle_edges - true_cycle_edges)
    fn = len(true_cycle_edges - est_cycle_edges)

    # Compute precision, recall, and F1-score
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return f1

def compute_cycle_SHD(B_true, B_est):
    """
    Compute the minimum column-permutation SHD for cyclic graphs.

    Parameters:
        B_true (np.ndarray): Ground truth adjacency matrix.
        B_est (np.ndarray): Estimated adjacency matrix.

    Returns:
        float: Minimum column-permutation SHD value.
    """
    return _compute_cycle_SHD(B_true, B_est)

def compute_min_KLD(prec_true, B_support, thresh=1e-6, max_iters=50, patience=10):
    """
    Compute the minimum Kullback-Leibler divergence (KLD) between a ground truth precision matrix and 
    an estimated precision matrix based on a given graph structure.

    This function attempts to find optimal parameters for a fixed graph structure that minimize the KLD
    with respect to the ground truth distribution. It serves as an evaluation metric for learned graph structures.

    Parameters
    ----------
    prec_true : numpy.ndarray
        Ground truth precision matrix of shape (d, d) where d is the number of variables
    B_support : numpy.ndarray
        Binary adjacency matrix of shape (d, d) representing the graph structure to evaluate
    thresh : float, optional
        Convergence threshold for KLD minimization. Default is 1e-6
    max_iters : int, optional
        Maximum number of optimization iterations. Default is 50
    patience : int, optional
        Number of consecutive non-improving iterations before early stopping. Default is 10

    Returns
    -------
    float
        Minimum KLD value achieved. A value close to zero indicates the estimated structure 
        is likely in the equivalence class of the true structure.

    Notes
    -----
    The function internally uses _minimize_kld which returns both the minimum KLD value and 
    the optimal Q matrix where Q @ Q.T = precision_estimated. This wrapper returns only the KLD value.
    """
    # _minimize_kld returns min/best KLD and the best Q matrix st Q @ Q.T = prec_est
    # here we only need the min KLD value, so we return only the first element of the tuple
    return _minimize_kld(prec_true, B_support, thresh, max_iters, patience)[0]

def compute_CSS(shd_cyclic, cycle_f1, epsilon=1e-8):
    """
    Compute the Cyclic Structural Score (CSS), which evaluates the learned graph structure
    by balancing the Cyclic Structural Hamming Distance (SHD) and the Cycle F1 score.

    The SHD used in this metric is computed relative to the best-case scenario for the 
    equivalence class of the true cyclic graph. This ensures that the comparison accounts for 
    structural variations that preserve the equivalence class, providing a fair evaluation.

    The CSS combines these two metrics to assess both structural accuracy and the ability
    to identify cycles. A perfect score is 1, with worse scores increasing above 1.

    Parameters:
        shd_cyclic (float): Cyclic Structural Hamming Distance (SHD) relative to the equivalence class.
        cycle_f1 (float): Cycle F1 score.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Cyclic Structural Score (CSS). A perfect score is 1, and worse scores are greater than 1.
    """
    return (shd_cyclic + 1) / (cycle_f1 + epsilon)

def compute_cycle_KLD(prec_mat_true, B_est, thresh=1e-6, max_iters=50, patience=10, Q_init: np.ndarray = None):

    """
        KLD-based evaluation of learned structure. Find parameters for a fixed graph structure (B_est) 
        in order to minimize KLD with the ground truth distribution (precision matrix prec_mat_true). 
        If B_est is in the equivalence class of the structure generating the precision matrix prec_mat_true, 
        then the minimum kld should be theoretically zero.

    Inputs:
        prec_mat_true: The precision matrix of the ground truth distribution.
        B_est: The estimated adjacency matrix.
        thresh: The threshold for convergence.
        max_iters: The maximum number of iterations.
        patience: The number of iterations to wait before early stopping.
        Q_init: Initial value Q such that Q @ Q.T = prec_mat_est. Q is a parameter matrix to construct the precision matrix.
        Note: Q is lower triangular matrix st Q @ Q.T is symmetric positive definite.

    Returns:
        float: Best KLD value.
        #np.ndarray: The best Q matrix that minimizes the KLD.
    """

    assert prec_mat_true.shape[0] == prec_mat_true.shape[1]
    dim = prec_mat_true.shape[0]

    (prec_sgn, prec_logdet) = np.linalg.slogdet(prec_mat_true)
    true_logdet = prec_sgn * prec_logdet
    prec_inv = np.linalg.inv(prec_mat_true)

    # objective function
    def _obj_kld(x):
        Q = x.reshape((dim, dim))
        prec_x = Q @ Q.T
        (x_sgn, x_uslogdet) = np.linalg.slogdet(prec_x)
        prec_x_logdet = x_sgn * x_uslogdet
        return 0.5 * (true_logdet - prec_x_logdet - dim + np.trace(prec_x @ prec_inv))

    # gradient for first order methods
    def _grad_kld(x):
        Q = x.reshape((dim, dim))
        return (-np.linalg.inv(Q).T + prec_inv @ Q).ravel()

    # make bounds based on supp matrix
    Q_support = (B_est + np.eye(dim)).astype(int)

    lb = [-np.inf if i != 0 else 0 for i in Q_support.ravel()]
    ub = [np.inf if i != 0 else 0 for i in Q_support.ravel()]

    best_KLD = np.inf
    Q_best = None
    wait = 0
    n_iters = 0
    while wait <= patience and best_KLD > thresh and n_iters <= max_iters:
        x0 = np.random.randn(dim**2)
        if n_iters == 0 and Q_init is not None:
            x0 = (Q_init * Q_support).ravel()
        res = minimize(
            _obj_kld, x0, jac=_grad_kld, method="L-BFGS-B", bounds=Bounds(lb, ub)
        )
        if res.fun < best_KLD:
            best_KLD = res.fun
            Q_best = res.x.reshape((dim, dim)).copy()
            wait = 0
        else:
            wait += 1

        n_iters += 1

    return best_KLD
    #return best_KLD, Q_best

#################################### MIXED DATA METRICS ######################################

def compute_TE(W, from_index, to_index):
    """
    Compute total effect from source to target using matrix inversion.
    
    Args:
        W (np.ndarray): Weighted adjacency matrix.
        from_index (int): Index of source variable (source).
        to_index (int): Index of target variable (target).
    
    Returns:
        float: Total effect from source to target.
    """
    try:
        I = np.eye(W.shape[0])
        T = np.linalg.inv(I - W) - I  # Compute total effect matrix
        return T[from_index, to_index]
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed, likely due to cycles (I - W is singular).")

def compute_TEE(W_true, W_est, from_index, to_index):
    """
    Compute Total Effect Estimation Error (TEE).

    Parameters:
    ----------
    W_true : np.ndarray
        Ground truth weighted adjacency matrix.
    W_est : np.ndarray
        Estimated weighted adjacency matrix.
    from_index : int
        Index of intervention/treatment/source variable.
    to_index : int
        Index of target/response variable.

    Returns:
    -------
    tuple
        (true_total_effect, est_total_effect, TEE):
        - true_total_effect: The true total effect based on B_true.
        - est_total_effect: The estimated total effect based on B_est.
        - TEE: The Total Effect Estimation Error (TEE).
    """
    # Compute the true total effect using compute_TE
    true_total_effect = compute_TE(W_true, from_index, to_index)

    # Compute the estimated total effect using compute_TE
    est_total_effect = compute_TE(W_est, from_index, to_index)

    # Compute the Total Effect Estimation Error (TEE)
    TEE = abs(est_total_effect - true_total_effect) / max(abs(true_total_effect), 1e-8)

    # Return all three values
    return true_total_effect, est_total_effect, TEE

def compute_SHD(B_true, B_est):
    """
    Compute Structural Hamming Distance (SHD) and normalised SHD.

    Parameters:
        B_true (np.ndarray): Ground truth adjacency matrix.
        B_est (np.ndarray): Estimated adjacency matrix.

    Returns:
        tuple: (normalised SHD, absolute SHD)
    """
    B_true_binary = (B_true > 0).astype(np.int8)
    B_est_binary = (B_est > 0).astype(np.int8)
    normalised_SHD, absolute_SHD = shd(B_true_binary, B_est_binary)
    return normalised_SHD, absolute_SHD

def compute_SID(B_true, B_est, edge_direction="from row to column"):
    """
    Compute Structural Intervention Distance (SID) using parent_aid.

    Parameters:
        B_true (np.ndarray): Ground truth adjacency matrix.
        B_est (np.ndarray): Estimated adjacency matrix.
        edge_direction (str): Edge direction format for gadjid.

    Returns:
        tuple: (normalised SID, absolute SID)
    """
    B_true_binary = (B_true > 0).astype(np.int8)
    B_est_binary = (B_est > 0).astype(np.int8)
    normalised_SID, absolute_SID = parent_aid(B_true_binary, B_est_binary, edge_direction)
    return normalised_SID, absolute_SID

def compute_ancestor_AID(B_true, B_est, edge_direction="from row to column"):
    """
    Compute Ancestor Adjustment Distance (AID).

    Parameters:
        B_true (np.ndarray): Ground truth adjacency matrix.
        B_est (np.ndarray): Estimated adjacency matrix.
        edge_direction (str): Edge direction format for gadjid.

    Returns:
        tuple: (normalised AID, absolute AID)
    """
    B_true_binary = (B_true > 0).astype(np.int8)
    B_est_binary = (B_est > 0).astype(np.int8)
    normalised_AID, absolute_AID = ancestor_aid(B_true_binary, B_est_binary, edge_direction)
    return normalised_AID, absolute_AID

############################ MISC  ############################

class CustomLingamObject(lingam.DirectLiNGAM):
    def set_adjacency_matrix(self, W):
        """Set the adjacency matrix manually for the LiNGAM model."""
        self._adjacency_matrix = W  # Manually set the adjacency matrix

def compute_causal_order(adjacency_matrix):
    """
    Compute the causal order (topological order) of nodes from an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix of the DAG.

    Returns
    -------
    list
        List of nodes in causal order (from roots to leaves).
    """
    # Convert the adjacency matrix to a directed graph
    G = nx.DiGraph(adjacency_matrix)
    return list(nx.topological_sort(G))

# this method is deprecated because it is based on lingam's internal implementation
def _compute_TE(X, W, from_index, to_index):
    """
    Compute the total effect of an intervention on a variable in a causal graph.

    Parameters
    ----------
    X : np.ndarray
        Observational data of shape (n_samples, n_features).
    W : np.ndarray
        Adjacency matrix of the causal graph, either binary or weighted.
    from_index : int
        Index of the intervention/treatment variable (source).
    to_index : int
        Index of the target/response variable.

    Returns
    -------
    float
        Estimated total effect from `from_index` to `to_index`.
    """
    # Validate inputs
    if W.shape[0] != W.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if X.shape[1] != W.shape[0]:
        raise ValueError("Number of features in X must match the size of the adjacency matrix.")

    # Create an instance of the custom LiNGAM object
    lingam_object = CustomLingamObject()

    # Set the adjacency matrix
    lingam_object.set_adjacency_matrix(W)

    # Compute and set the causal order
    causal_order = compute_causal_order(W)
    lingam_object._causal_order = causal_order

    # Estimate and return the total effect
    total_effect = lingam_object.estimate_total_effect(X=X, from_index=from_index, to_index=to_index)
    return total_effect
