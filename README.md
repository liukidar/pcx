# PCAX

##  Introduction

pass

## Install

First, create an environment with Python 3.10 or newer and install JAX in the correct version for your accelerator device. Then

### Install stable version

On the right side of the repository, click on "releases" and download the wheel file. You can install it using

TODO: add wheel file install script

```shell
pip install ?
```

### Install from github

clone this repository locally and then:

```shell
pip install -e /path/to/this/repo/
```

To download the documentation in the `docs` folder, run:

```shell
git submodule update --init --recursive
```

If the `docs` folder remains empty, you probably do not have access to the [pcax-docs](https://github.com/liukidar/pcax-docs) repository yet.

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

## Contribute

TODO add contribute guidelines:

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
