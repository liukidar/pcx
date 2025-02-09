import os
import pandas as pd
import numpy as np
import networkx as nx

# Import utility functions
from cyclic_obs_data_generator import sample_ER_dcg, sample_SF_dcg, sample_NWS_dcg
from cyclic_obs_data_generator import sample_W, sample_data

def regenerate_missing_datasets(data_dir, num_samples_list, noise_type="GAUSS-EV"):
    """
    Identify missing datasets and regenerate them using the stored weighted adjacency matrix.

    Args:
        data_dir (str): Root directory containing dataset folders.
        num_samples_list (list): List of sample sizes that should exist.
        noise_type (str): Noise type used in dataset generation.
    """
    # Iterate over all dataset folders
    for dataset_folder in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_folder)

        # Ensure it's a valid directory
        if not os.path.isdir(dataset_path):
            continue  

        W_path = os.path.join(dataset_path, "W_adj_matrix.csv")
        prec_path = os.path.join(dataset_path, "prec_matrix.csv")

        # Check if necessary files exist
        if not os.path.exists(W_path) or not os.path.exists(prec_path):
            print(f"⚠️ Skipping {dataset_folder}, missing W_adj_matrix.csv or prec_matrix.csv.")
            continue

        # ✅ Load weighted adjacency matrix
        W = pd.read_csv(W_path, header=None).values

        # ✅ Load precision matrix (not used for dataset generation but ensuring it's present)
        prec_matrix = pd.read_csv(prec_path, header=None).values

        # ✅ Check for missing datasets
        for num_samples in num_samples_list:
            dataset_subdir = os.path.join(dataset_path, f"n_samples_{num_samples}")
            train_file = os.path.join(dataset_subdir, "train.csv")

            if not os.path.exists(train_file):  # Only regenerate missing ones
                print(f"🛠️ Regenerating dataset for {dataset_folder}, n_samples={num_samples}...")

                # Generate new data
                d = W.shape[0]  # Number of variables
                scales = np.ones(d, dtype=float)
                X_full, _ = sample_data(W=W, scales=scales, num_samples=num_samples, noise_type=noise_type)

                # Convert to DataFrame and create directory if needed
                os.makedirs(dataset_subdir, exist_ok=True)
                pd.DataFrame(X_full).to_csv(train_file, header=False, index=False)

    print("✅ Missing dataset generation complete.")

data_dir = "/share/amine.mcharrak/cyclic_data_final"
num_samples_list = [500, 1000, 2000, 5000]  # Ensure all sample sizes are checked

regenerate_missing_datasets(data_dir, num_samples_list)
