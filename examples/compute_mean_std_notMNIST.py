import os
import numpy as np
import deeplake

from causal_helpers import set_random_seed

# Set random seed for reproducibility
set_random_seed(42)

# Load notMNIST dataset from deeplake
ds = deeplake.load('hub://activeloop/not-mnist-small')

# Get the dataloader without shuffling
dataloader = ds.pytorch(num_workers=16, batch_size=len(ds), shuffle=True)

# Fetch the images and targets from the dataloader
batch = next(iter(dataloader))
images, targets = batch['images'], batch['labels']

# Convert images and targets to numpy arrays
images_np = images.numpy()
targets_np = targets.numpy().astype(np.uint8).squeeze()

# Ensure the images have the correct shape
images_np = images_np.reshape(-1, 28, 28)

# Convert the images to the range [0.0, 1.0]
images_np = images_np / 255.0

# Compute mean and std on a random subset of the images of size 50%
subset_size = len(images_np) // 2
images_subset = images_np[:subset_size]

# Calculate the mean and std
mean, std = np.mean(images_subset), np.std(images_subset)

# Print the mean and std
print("Mean of notMNIST dataset:", mean)
print("Std of notMNIST dataset:", std)

# NOTE results are:
# Mean of notMNIST dataset: 0.42479814506447755
# Std of notMNIST dataset: 0.45854366793268586