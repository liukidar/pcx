# PCAX

##  Introduction

pass

## Todo

- environments.yaml should be requirements.txt
- Add licence
- Protect Main Branch
- Add auto changelog
- Add pre-commit hooks
- Add tests
- Add docs as submodule
- Set Merge to "Squash and Merge"

## Install

1) Create conda environment
2)

```shell
pip install -e /path/to/this/repo/pcax
```

## Contribute

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
