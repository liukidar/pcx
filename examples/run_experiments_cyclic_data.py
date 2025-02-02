import subprocess
import time

# Define the list of seeds
seeds = [1, 2, 3, 4, 5]

# Create a list to store the processes
processes = []

for seed in seeds:
    print(f"Starting experiment with seed {seed}...")
    with open(f"experiment_results/experiment_seed_{seed}.log", "w") as log_file:
        p = subprocess.Popen(
            ["python", "icml_cyclic_data_script.py", "--seed", str(seed)],
            stdout=log_file, stderr=log_file
        )
        processes.append(p)

# Wait for all processes to complete
while any(p.poll() is None for p in processes):
    print("Waiting for experiments to finish...")
    time.sleep(5)

print("All experiments finished.")
