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
WORKDIR /home/pcax/workspace

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

# https://stackoverflow.com/a/31532674/4119680
COPY --chown=pcax:pcax pyproject.toml poetry.loc[k] ./

RUN poetry install --no-root

