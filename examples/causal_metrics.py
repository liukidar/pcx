import numpy as np
import networkx as nx

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

def f1_skeleton(B_true, B_est):
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

def f1_directed(B_true, B_est):
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

def compute_auroc(B_true, W_est, n_points=50):
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

def compute_auprc(B_true, W_est, n_points=50):

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

def cycle_f1(B_true, B_est):
    """
    Compute the F1 score based on all edges involved in cycles in B_true and B_est.

    Parameters:
        B_true (np.ndarray): The true directed cyclic graph.
        B_est (np.ndarray): The estimated directed cyclic graph.

    Returns:
        float: The F1 score based on cycle-related edges.
    """
    # Convert input arrays to directed graphs
    G_true = nx.DiGraph(B_true)
    G_est = nx.DiGraph(B_est)

    # Ensure B_true contains cycles
    true_cycles = list(nx.simple_cycles(G_true))
    if not true_cycles:
        raise ValueError("B_true is not cyclic. Please provide a cyclic true graph.")

    # Ensure B_est contains cycles
    est_cycles = list(nx.simple_cycles(G_est))
    if not est_cycles:
        return 0.0  # No cycles in B_est means 0 F1-score

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
