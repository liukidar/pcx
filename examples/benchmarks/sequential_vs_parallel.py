import os
import time
import numpy as np
import re
import tqdm

TRIALS = 10
BEST_OF = 5
COOLDOWN = 5

# Uses either mnist_pexec.py or mnist.py
file = "mnist_pexec.py"
os.remove("sequential_vs_parallel.txt")
for i in tqdm.tqdm(range(TRIALS)):
    t = os.system(f"python ../{file} >> sequential_vs_parallel.txt")
    time.sleep(COOLDOWN)

with open("sequential_vs_parallel.txt", "r") as f:
    content = f.read()

epoch_times = re.findall(r'An Epoch takes on average ([\d.]+) seconds', content)
epoch_times = np.array([float(time) for time in epoch_times])

mean_time = np.mean(epoch_times)
print(f"Mean time ({TRIALS} times): ", mean_time.item())
best_times = np.argsort(epoch_times)[:BEST_OF]
print("Mean best 5 times: ", np.mean(epoch_times[best_times]).item())
