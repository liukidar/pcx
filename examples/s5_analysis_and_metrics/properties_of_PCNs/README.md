# Properties of Predictive Coding Networks: Experiment Replication

This repository contains the code to replicate the experiments described in Section 5.2 on Properties of Predictive Coding (PC) networks.

## Overview

In this tutorial, we analyze how the energy in a Multi-Layer Perceptron (MLP) neural network, trained with Predictive Coding (PC), can be used to assess whether test data belongs to in-distribution or out-of-distribution. The energy of the network is used as a measure of surprise.

## Steps to Replicate the Experiment

1. **Prepare the Datasets:**

    - In-distribution: MNIST
    - Out-of-distribution: FashionMNIST, notMNIST, KMNIST, EMNIST (letters)
    - Note on interpolation threshold: For a regular train set size of (n=60000) with K=10, the interpolation threshold (I_th) is 600,000 samples.
    - To fit a model in the "modern" over-parameterized regime, we need a 3-layer-MLP with 800 hidden nodes (h=800), having approximately 1.2x10^6 parameters, which is twice as big as the interpolation threshold.
    - Using a subset of the training data (n_subset_size=30000) reduces the interpolation threshold to 300,000 samples. For K=10 classes, a subset size of 30,000 samples requires a hidden size h≈512, resulting in a neural network with approximately 650,000 parameters.

2. **Training Phase:**

    - Train a MLP neural network with Predictive Coding (PC) on the MNIST dataset.

3. **Testing Phase:**

    - Test the trained network using both MNIST and FashionMNIST datasets.

4. **Visualization and Analysis:**

    - **Energy Distribution Histograms:** Visualize the distribution and spread of the energy for in-distribution and out-of-distribution data using separate histograms for each dataset, and overlay them for comparison.
    - **Box Plots:** Summarize the central tendency and variability of energies using box plots to show the median, quartiles, and outliers.
    - **ROC Curve:** Generate a Receiver Operating Characteristic (ROC) curve based on the energies to assess the model's ability to distinguish between in-distribution and out-of-distribution data. Plot true positive rate (sensitivity) against false positive rate (1-specificity) for different threshold values of energies.
    - **Scatter Plot:** Create a scatter plot of energies versus prediction confidence (argmax softmax values) for both MNIST and FashionMNIST to investigate the relationship between model confidence and energy value, using different colors or markers for MNIST and FashionMNIST data points.

5. **Likelihood Computation:**
    - Compute the likelihoods of various datasets and compare them against in-distribution and out-of-distribution data samples.

## How to Run the Code

1. **Train the Model:**

    - To train the model on MNIST and infer energy values and predictions for all datasets (including notMNIST, EMNIST, KMNIST, and FashionMNIST), run the `OOD_energy.py` script. This script also stores the data in pickle format for later analysis.

    ```bash
    python OOD_energy.py
    ```

2. **Generate Plots:**

    - To create the plots from the paper, run the Jupyter Notebook file `energy_neurips.ipynb` from within the same folder.

    ```bash
    jupyter notebook energy_neurips.ipynb
    ```

## Additional Information

-   **Helper Functions:** The `utils.py` file contains additional helper functions for the main training and OOD analysis scripts.

## Reproducibility

To ensure reproducibility, we provide the `environment.yml` file to set up the Conda environment:

```bash
conda env create -f environment.yml
```
