# PCX -- Predictive Coding Networks Made Simple

## Introduction

PCX is a Python JAX-based library designed to develop highly configurable predictive coding networks. Please refer to the tutorial notebooks in the examples folder to get started. PCX can be installed by following one of the listed three methods.

## Default: Installation via PIP [Method #1].

First, create an environment with Python >= 3.10 and [install JAX](https://github.com/google/jax#installation) in the correct version for your accelerator device. For cuda >= 12.0, the command is

```shell
pip install -U "jax[cuda12]"
```

For CPU only:

```shell
pip install -U "jax[cpu]"
```

Then you hav two options:

-   Install a stable version
-   Clone this repository and install the package by linking to the this folder. The installation of this libary only links to this folder and thus dynamically updates with all your changes.

### Install stable version

On the right side of the repository, click on "releases" and download the wheel file. You can install it using

```shell
pip install path/to/wheel_file.whl
```

Alternatively you can use the PyPi version by [work in progress...]

### Install dynamically from github

Clone this repository locally and then:

```shell
pip install -e /path/to/this/repo/ --config-settings editable_mode=strict
```

## Ensuring Reproducibility: Installation via `poetry` [Method #2]

**TL;DR** This is an alternative installation method that creates a fully configured environment to ensure your results are reproducible (no pip install, see previous section for that; no docker install, see the next section for docker install):

1. Install [conda](https://www.anaconda.com/).
2. Install [poetry](https://python-poetry.org/).
3. `poetry config virtualenvs.create false`.
4. Create a conda environment with python>=3.10: `conda create -n pcax python=3.10`.
5. Activate the environment: `conda activate pcax`.
6. `cd` into the root pcax folder.
7. `poetry install --no-root`.

In this way, we use [poetry](https://python-poetry.org/) to make sure the environment is 100% reproducible. If you are not familiar with `poetry`, now is a good time to skim through the docs.

### Development Notes:

1. If you need to add a Python package to the environment, use `poetry add package`. Avoid `pip install`!
2. If you want to update a version of an existing package, run `poetry update package`. It will update the package to the latest available version that fits the constraints.
3. **DO NOT** update the package versions in the `pyproject.toml` file manually. Surprisingly, `pyproject.toml` **DOES NOT** specify the versions that will be installed, `poetry.lock` does. So, first check the package version in `poetry.lock`.
4. **DO NOT** update the package versions in the `poetry.lock` file manually. Use `poetry update package` instead. `poetry.lock` **HAS** to be generated and signed automatically.
5. If `pyproject.toml` and `poetry.lock` have diverged for some reason (for example, you've merged another branch and resolved conflicts in `poetry.lock`), use `poetry lock --no-update` to fix the `poetry.lock` file.
6. **DO NOT** commit changes to `pyproject.toml` without running `poetry lock --no-update` to synchronize the `poetry.lock` file. If you commit `pyproject.toml` that is not in sync with `poetry.lock` this will break the automatic environment configuration for everyone.

## Fully Automatic: Environment in Docker with Dev Containers [Method #3]

Run your development environment in a docker container. This is the most straightforward option to work with `pcx`, as the development environment is pre-configured for you.

The `Dockerfile` is located in `pcx/docker`, with the `run.sh` script that builds and runs it. You can play with the `Dockerfile` directly if you know what you are doing or if you don't use VSCode. If you want a fully automated environment setup, then forget about the `pcx/docker` directory and read on.

**Warning**: This image should run on CUDA 12.2 or later, but not earlier. Make sure that your `nvidia-smi` reports CUDA >=12.2. If not, update the base `nvidia/cuda` image and the fix at the bottom in the `docker/Dockerfile` to use the same version of CUDA as your host does.

Requirements:

1. A CUDA >=12.2 enabled machine with an NVIDIA GPU. You can do without a GPU, probably, just omit the steps related to the GPU passthrough and configuration.
2. [Install docker](https://docs.docker.com/engine/install/).
3. Install [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to enable docker to use the GPU.
4. **Make sure to re-start the docker daemon after the previous step**. For example, on Ubuntu this will be `sudo systemctl restart docker`.
5. Install [Visual Studio Code](https://code.visualstudio.com/download).
6. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VSCode.
7. Optionally, [read how to develop inside container with VS Code](https://code.visualstudio.com/docs/devcontainers/containers).

Once everything is done, open this project in VS Code and execute the `Dev Containers: Reopen in Container` command (Ctrl/Cmd+Shift+P). This will build the docker image and open the project inside that docker image. Building the docker image for the first time may take around 15-30 minutes, depending on your internet speed.

You can always exit the container by running the `Dev Containers: Reopen folder locally` command.
You can rebuild the container by running the `Dev Containers: Rebuild Container` command.

You can check that you're running inside a container by running `hostname`. If it outputs meaningless 12 characters, then you are inside a container. If it outputs the name of your machine, you are not in a container.

When running a Jupyter Notebook it will prompt you to select an environment. Select Python Environments -> Python 3.10 (any of them, as they are the same).

**Important notes**:

- You are not supposed to modify the `docker/Dockerfile` unless you perfectly know what you are doing and why.
- You are not supposed to run the docker container directly. The Dev Containers extension will do this for you. If you think you need to `docker run -it` then something is really wrong.
- Use `poetry` to add a python package to the environment: `poetry add --group dev [package]`. The `--group dev` part should be omitted if this package is needed for the core `pcx` code. Try not to install packages with `pip`.
- Please update your docker to >>20.10.9. [This image is known not to work with docker <= 20.10.9](https://stackoverflow.com/questions/71941032/why-i-cannot-run-apt-update-inside-a-fresh-ubuntu22-04). It failes with the following message: `E: Problem executing scripts APT::Update::Post-Invoke 'rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true'`.
- Sometimes Pylance fails to start because it depends on the Python extension that starts later. In this case, just reload the window by running the `Developer: Reload window` command.

**PyTorch with GPU support**: By default, the image will install a CPU-only PyTorch. If you need GPU support with PyTorch, do the following:

1. Open the project in a container using DevContainers as described above.
2. Replace ALL occurrences of `source = "torch-cpu"` with `source = "torch-gpu"` in the [pyproject.toml](./pyproject.toml) file.
3. Run `poetry lock --no-update` to re-generate the `poetry.lock` file. Note that you should do it while running inside the container.
4. Run `poetry install`. Make sure you run it inside the container. It will take up to 20 minutes.

## Citation
If you found this library to be useful in your work, then please cite: [arXiv link](https://arxiv.org/abs/2407.01163)

```bibtex
@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple}, 
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163}, 
}
```

## Documentation

The documentation is available at: [https://pcx.readthedocs.io/en/stable/](https://pcx.readthedocs.io/en/stable/)

To learn how to build it yourself, go to `/docs/README.md`.

## Contributing

If you want to contribute to the project, please read [CONTRIBUTING.md](CONTRIBUTING.md)
