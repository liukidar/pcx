import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


import numpy as np


def sample_param_from_B(B_given, B_low=0.5, B_high=2.0, noise_low=0.75, noise_high=1.25,
                        max_eig=1.0, max_cond_number=20, max_attempts=1000):
    """
    Generate graph parameters given a custom adjacency matrix B_given,
    where entries of B_given represent positive integer strengths.
    Assign random signs to the parameters and resample signs and scale down
    absolute values during stability checks.
    Returns B_sampled and noise scales.
    """
    assert B_given.shape[0] == B_given.shape[1], "B_given must be a square matrix."
    n_variables = B_given.shape[0]
    
    # Get indices of non-zero entries
    nonzero_indices = np.where(B_given > 0)
    
    # Map B_given[i,j] to absolute b_ij in [B_low, B_high], preserving the ordering
    min_B_given = np.min(B_given[nonzero_indices])
    max_B_given = np.max(B_given[nonzero_indices])
    B_abs_sampled = np.zeros_like(B_given, dtype=float)
    
    # Avoid division by zero if min_B_given == max_B_given
    if min_B_given == max_B_given:
        # All non-zero entries in B_given have the same value
        # Set all |b_ij| to B_low
        B_abs_sampled[nonzero_indices] = B_low
    else:
        # Linear mapping to [B_low, B_high]
        B_abs_sampled[nonzero_indices] = ((B_given[nonzero_indices] - min_B_given) / (max_B_given - min_B_given)) \
                                         * (B_high - B_low) + B_low
    
    # Initialize scaling factor
    scaling_factor = 1.0
    
    # Initialize signs
    signs = np.ones_like(B_given)
    
    # Stability check loop
    stable = False
    attempt = 0
    while not stable and attempt < max_attempts:
        # Resample signs
        signs[nonzero_indices] = np.random.choice([-1, 1], size=len(nonzero_indices[0]))
        
        # Scale down absolute values
        scaling_factor *= 0.9  # Reduce the scaling factor
        B_abs_scaled = B_abs_sampled * scaling_factor
        
        # Apply signs
        B_sampled = B_abs_scaled * signs
        
        # Check stability
        eigvals = np.linalg.eigvals(B_sampled)
        if np.max(np.abs(eigvals)) < max_eig and np.linalg.cond(np.eye(n_variables) - B_sampled) < max_cond_number:
            stable = True
        else:
            attempt += 1
    
    if not stable:
        raise ValueError("Failed to find a stable B matrix within max attempts")
    
    # Sample noise scales
    noise_scales = np.random.uniform(noise_low, noise_high, size=n_variables)
    
    return B_sampled, noise_scales


def sample_graph(B, noise_params, n_samp, noise_type='gaussian'):
    """
    Generate data given B matrix and noise parameters.
    noise_type: 'gaussian' or 'non-gaussian'
    """
    p = len(noise_params)
    
    if noise_type == 'gaussian':
        # Gaussian noise
        N = np.random.normal(0, np.sqrt(noise_params), size=(n_samp, p))
    elif noise_type == 'non-gaussian':
        # Non-Gaussian noise
        noise = [np.random.uniform(-scale, scale, size=(n_samp)) for scale in noise_params]
        N = np.array(noise).T
        N = N ** 5  # Introduce non-Gaussianity
    else:
        raise ValueError("Invalid noise_type. Must be 'gaussian' or 'non-gaussian'")
    
    return (np.linalg.inv(np.eye(p) - B.T) @ N.T).T


def sample_cyclic_data(B_given, n_samples, B_low=0.5, B_high=2.0, noise_low=0.75, noise_high=1.25,
                       max_eig=1.0, max_cond_number=20, max_attempts=1000, noise_type='gaussian'):
    """
    Generate data from a custom adjacency matrix B_given with cycles.
    Returns data X of shape (n_samples, n_variables) and the sampled B matrix.
    noise_type: 'gaussian' or 'non-gaussian'
    """
    B_sampled, noise_params = sample_param_from_B(
        B_given, B_low=B_low, B_high=B_high, noise_low=noise_low, noise_high=noise_high,
        max_eig=max_eig, max_cond_number=max_cond_number, max_attempts=max_attempts
    )
    X = sample_graph(B_sampled, noise_params, n_samples, noise_type=noise_type)
    return X, B_sampled


# Example usage:
if __name__ == "__main__":
    # Define your custom adjacency matrix B_given (positive integers)
    B_given = np.array([
        [0, 40, 0, 0],
        [0, 0, 30, 0],
        [0, 0, 0, 20],
        [10, 0, 0, 0]
    ])
    
    n_samples = 10000
    
    # For Gaussian noise
    X_gaussian, B_sampled_gaussian = sample_cyclic_data(
        B_given, n_samples, noise_type='gaussian'
    )
    print("Generated data with Gaussian noise, shape:", X_gaussian.shape)
    print("Sampled B matrix with Gaussian noise:\n", B_sampled_gaussian)
    
    # For Non-Gaussian noise
    X_nongaussian, B_sampled_nongaussian = sample_cyclic_data(
        B_given, n_samples, noise_type='non-gaussian'
    )
    print("Generated data with Non-Gaussian noise, shape:", X_nongaussian.shape)
    print("Sampled B matrix with Non-Gaussian noise:\n", B_sampled_nongaussian)
    
    # use networkx to plot the graph and print whether it is acyclic or cyclic and whether it is directed
    G = nx.from_numpy_array(B_sampled_nongaussian, create_using=nx.DiGraph)
    print("Is the graph acyclic?", nx.is_directed_acyclic_graph(G))
    print("Is the graph directed?", nx.is_directed(G))
    nx.draw(G, with_labels=True)
    # make plot directory if it does not exist
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    plt.savefig('../plots/demo_B_sampled_graph.png')
    plt.show()