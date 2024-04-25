# https://hub.docker.com/r/nvidia/cuda/tags
# Select the tag that matches the CUDA version on the host.
# Make sure to select a CuDNN image!
# See the compatibility matrix here: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package
# Also make sure to change the fix at the very end of this file.
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Install Python 3.10 and pip3
RUN apt update \
    && apt install --fix-missing -y python3 python3-pip curl vim less openssh-client \
    && ln -s $(which python3) /usr/bin/python

# Switch to a non-root user
# Dev containers should substitute for the current user on startup but that doesn't work: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd -g 1000 pcax \
    && useradd -m -u 1000 -g pcax pcax \
    # Allow the user to install packages
    && chmod -R a+rwx /usr/local /usr/include/python3.10
USER pcax:pcax

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="$PATH:/home/pcax/.local/bin:" \
    # Disable keyring so poetry doesn't get stuck
    # See this issue: https://github.com/python-poetry/poetry/issues/5250
    # and this PR: https://github.com/python-poetry/poetry/pull/5251
    PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
    # Prevent JAX from allocating 90% of GPU memory:
    XLA_PYTHON_CLIENT_PREALLOCATE="false"

RUN poetry config virtualenvs.create false

# RUN pip3 install "torch>=2.2.2" torchvision umap-learn hyperparameters

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

# # Important: Make sure to install JAX last:
# # 1. JAX cannot work with PyTorch's CUDA while the opposite works. Thus, JAX has to overwrite the CUDA in the environment.
# # 2. poetry will rollbacl jaxlib to a non-CUDA version. Thus, the command below must be run after any env changes made by poetry.
# RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # Fix the warning with newer CUDA installed by JAX than the driver's CUDA version:
# # W external/xla/xla/service/gpu/nvptx_compiler.cc:718] The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.4.131). Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.
# # This happens because the nvidia-cuda-nvcc-cu12 >= 12.4 gets installed by JAX but this docker image is based on CUDA 12.2.
# # See this issue: https://github.com/google/jax/issues/18027
# # Remove this fix if you are using a newer CUDA version.
# RUN pip install "nvidia-cuda-nvcc-cu12>=12.2,<12.3"


