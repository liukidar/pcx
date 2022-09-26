# PCAX

##  Introduction

TODO add introduction to the libary.

## Install

First, create an environment with Python 3.10 or newer and install JAX in the correct version for your accelerator device. Then you hav two options: a) Install a stable version or b) clone this repository and install the package by linking to the this folder. The installation of this libary only links to this folder and thus dynamically updates all your changes.

### A Install stable version

On the right side of the repository, click on "releases" and download the wheel file. You can install it using

```shell
pip install path/to/wheel_file.whl
```

### B Install from github

Clone this repository locally and then:

```shell
pip install -e /path/to/this/repo/
```

### Docs

To download the documentation in the `docs` folder, run:

```shell
git submodule update --init --recursive
```

If the `docs` folder remains empty, you probably do not have access to the [pcax-docs](https://github.com/liukidar/pcax-docs) repository yet.

This only contains the raw doc files and not the compiled folders. If you need to view the docs, you can compile them. #TODO

## Todo

- Add licence and then change it under setup.cfg licence
- Add auto changelog
- Add pre-commit hooks
- Add tests
- Add docs as submodule
- Set Merge to "Squash and Merge"
- Add submodule docs in packaging
- GHA fail on formatting / linting issue
- Update Mypy as soon as <https://github.com/python/mypy/issues/13627#issuecomment-1240582303> is fixed
- add a GHA for the docs repo to automatically create a PR in this repo to update the submodule. [This GHA might help](https://github.com/releasehub-com/github-action-create-pr-parent-submodule).

## Contribute

Please use a new branch to contribute. Once you are done writing the feature, please open a Pull Request. A few github actions will be triggered: Formatting, Linting, Type Checking and Testing. These steps ensure that the library maintains a certain level of quality and fascilitate collaboration.

- Formatting: We use `black` to format the code. Check out [pyproject.toml](pyproject.toml) for the arugments. This ensures that all files from all authors are coherently formatted.
- Linting: This checks the code for style quality using `flake8` and `isort`. `flake8` complains about style errors, as for example variable names that are too generic or unused imports. `isort` complains if the imports within each block are not alphabetically ordered. For linting arguments check [setup.cfg](setup.cfg).
- Type Checking: This checks if type hints match the types that are actually passed through the code using `mypy`. This isnt infallable but it helps a lot making the typing more explicit. If you cant find a solution to satisfy `mypy` you can mark that line with a `# noqa: typing` comment and `mypy` will ignore errors in this line. For arguments check [pyproject.toml](pyproject.toml).
- Testing: All code in the library should be explicitly tested. We use `pytest` to do that. Please add tests for any feature you implement.

The GHA only check that everything is in order, but does not change the code for you. Please, do the formatting, testing etc locally. You can use the logs of the GHA to see where it is failing. Once all the GHA pass, request Luca as a reviewer for your PR. Once he approves, you should use the `Squash and Merge` functionalty to merge your feature on the main branch. `Squash and Merge` means that all your little commits on the feature branch will be bundles into one big commit.

Please add comments and docstrings to your changes.

One warning: We cannot test GPU features on GHA. Please do this locally as well, even though it might not result in an error in the GHA.

TODO How do update the documentation?

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
