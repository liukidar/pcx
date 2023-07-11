# Installation

First, create a Conda environment with Python 3.10 and activate it:
```
conda create -n pcd python=3.10
conda activate pcd
```

Install Nightly PyTorch (for CUDA 12.1 or above) as [described here](https://pytorch.org/):
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
Note that we are using the Nightly build to get support for CUDA 12.
Make sure to install JAX after installing PyTorch, as both of them install their own CUDA version, and JAX cannot work with PyTorch's CUDA while the opposite works.

Now you can use `poetry` to install the rest of dependencies:
```
poetry install --with dev
```

There are two packages that still have to be installed with `pip`:
```
pip install umap-learn hyperparameters
```

Important: Make sure to install JAX last:
1. JAX cannot work with PyTorch's CUDA while the opposite works. Thus, JAX has to overwrite the CUDA in the environment.
2. `poetry` will rollbacl `jaxlib` to a non-CUDA version. Thus, the command below must be run after any env changes made by `poetry`.

Next, install JAX and CUDA toolkit as [described here](https://github.com/google/jax#installation):
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
This will install CUDA toolkit for CUDA 12.2. If you have older CUDA, use the following env variable for compatibility:
```
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
```

To prevent JAX from allocating 90% of GPU memory on start, set the following environment variable:
```
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
```
