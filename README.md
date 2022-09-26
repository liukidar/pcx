# PCAX

## Introduction

pcax is a Python JAX-based library designed to develop highly configurable predictive coding networks. It is strictly forbidden to share any piece of code without permission.

## Install

First, create an environment with Python 3.10 or newer and [install JAX](https://github.com/google/jax#installation) in the correct version for your accelerator device. For cuda >= 11.4, the command is

```shell
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then you hav two options:

- Install a stable version
- Clone this repository and install the package by linking to the this folder. The installation of this libary only links to this folder and thus dynamically updates all your changes.

### Install stable version

On the right side of the repository, click on "releases" and download the wheel file. You can install it using

```shell
pip install path/to/wheel_file.whl
```

### Install dynamically from github

Clone this repository locally and then:

```shell
pip install -e /path/to/this/repo/
```

## Docs

To download the documentation in the `docs` folder, run:

```shell
git submodule update --init --recursive
```

If the `docs` folder remains empty, you probably do not have access to the [pcax-docs](https://github.com/liukidar/pcax-docs) repository yet.

The documentation does not need to be compiled and you can view it from the [docs/index.html](docs/index.html) file. In the future, we will host the docs of the latest commit on the main branch on PSSR. Link will follow.

## Contribute

Please use a new branch to contribute. Once you are done writing the feature, please open a Pull Request. A few github actions will be triggered: Formatting, Linting, Type Checking and Testing. These steps ensure that the library maintains a certain level of quality and fascilitate collaboration.

- Formatting: We use `black` to format the code. Check out [pyproject.toml](pyproject.toml) for the arguments. This ensures that all files from all authors are coherently formatted.
- Linting: This checks the code for style quality using `flake8` and `isort`. `flake8` complains about style errors, as for example variable names that are too generic or there exist unused imports. `isort` complains if the imports within each block are not alphabetically ordered. For linting arguments check [setup.cfg](setup.cfg).
- Type Checking: This checks if type hints match the types that are actually passed through the code using `mypy`. This isnt infallable but it helps a lot making the typing more explicit. If you can't find a solution to satisfy `mypy` you can mark that line with a `# type: ignore` comment and `mypy` will ignore errors in this line. For arguments check [pyproject.toml](pyproject.toml).
- Testing: All code in the library should be explicitly tested. We use `pytest` to do that. Please add tests for any feature you implement.

The GHA only check that everything is in order, but does not alter the code for you. Please, do the formatting, testing etc locally. You can use the logs of the GHA to see where it is failing. Once all the GHA pass, request Luca as a reviewer for your PR. Once he approves, you should use the `Squash and Merge` functionalty to merge your feature on the main branch. `Squash and Merge` means that all your little commits on the feature branch will be bundles into one big commit and keep the commit tree tidy.

Please add comments and docstrings to your changes.

One warning: We cannot test GPU features on GHA. Please do this locally as well, even though it might not result in an error in the GHA.

### VSCode

This might be useful when using vscode:
Add this to you `settings.json`:

```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length 120"
    ]
}
```

It will continuously check your code using flake8 and mypy and set the formatting to black. As the linting is expensive, creating pre-commit hooks would be nicer though.

### Skip Github Actions

If you want to skip github actions on a single commit (e.g. an intermediate commit or comment that does not need to be checked), you can start your commit message with `[skip ci]`.
