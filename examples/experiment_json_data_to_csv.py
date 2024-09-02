import os
import json
import pandas as pd

# Define the path to the experiments folder
experiments_path = "/users-2/amine/pcax24/examples/experiments"  # Update this path if necessary

# List of graph types to look for
graph_types = ["ER", "SF", "connectome"]

# Initialize data storage for each graph type
data = {graph_type: [] for graph_type in graph_types}

# Iterate through all folders in the experiments directory
for folder in os.listdir(experiments_path):
    folder_path = os.path.join(experiments_path, folder)
    
    # Check if the current item is a directory and contains a relevant graph type
    # Also check if it contains a "model" subfolder indicating training completion
    if os.path.isdir(folder_path) and any(graph in folder for graph in graph_types) and os.path.isdir(os.path.join(folder_path, "model")):
        # Identify the graph type
        graph_type = next((graph for graph in graph_types if graph in folder), None)
        
        # Load hyperparams.json
        hyperparams_path = os.path.join(folder_path, "hyperparams.json")
        if not os.path.exists(hyperparams_path):
            continue
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        
        # Load metrics.json
        metrics_path = os.path.join(folder_path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Extract the last values of MAEs, SHDs, and energies if they exist
        last_mae = metrics['MAEs'][-1] if 'MAEs' in metrics and metrics['MAEs'] else None
        last_shd = metrics['SHDs'][-1] if 'SHDs' in metrics and metrics['SHDs'] else None
        last_energy = metrics['energies'][-1] if 'energies' in metrics and metrics['energies'] else None
        
        # Combine hyperparameters and metrics into one dictionary
        combined_data = {**hyperparams, 'MAE': last_mae, 'SHD': last_shd, 'energy': last_energy}
        
        # Append to the relevant graph type list
        data[graph_type].append(combined_data)

# Create and save CSV files for each graph type
for graph_type, records in data.items():
    if records:
        df = pd.DataFrame(records)
        csv_filename = f"{graph_type}_experiments.csv"
        csv_path = os.path.join(experiments_path, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_filename} with {len(records)} records.")