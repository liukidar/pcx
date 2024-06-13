# Examples

-   PCAX comes with examples that showcase how to implement different architectures using the library.
-   Examples also include hypertuning code, search spaces, and the best parameters, which makes it easy to experiment and reproduce the results from the paper.
-   Both PC and BP versions of architectures are provided for performance and implementation comparison.

## Common setup

Python dependencies for all examples are specified in the `examples` group in the [pyproject.toml](../pyproject.toml) file. This dependency group is installed automatically when you run `poetry install`. However, if you installed dependencies without the `examples` group, make sure to install it:

1. Activate the virtual environment: `conda activate pcax`.
1. `cd` in the root directory of `pcax`.
1. `poetry install --no-root --with examples`

If you want to run hypertuning, you will also need to install [stune](https://github.com/liukidar/stune/blob/v0.1.0/README.md):

1. `cd` outside of the `pcax` root directory.
1. Clone the [stune](https://github.com/liukidar/stune) repository: `git clone git@github.com:liukidar/stune.git`.
1. `cd stune`
1. Activate the virtual environment: `conda activate pcax`.
1. `pip install -e .`
1. Install additional stune dependencies: `pip install omegaconf-argparse names_generator redis optuna-dashboard`
1. `cd` into the `pcax/examples` folder.
1. **In a new terminal**, launch Redis. For this, you need to install [podman](https://podman.io/) or [docker](https://www.docker.com/): `./run_redis.sh`. Return to the original terminal after this step.
1. **You should do this step for each example subfolder!**. Configure stune in `pcax/examples/autoencoders/deconv` folder: `cd pcax/examples/autoencoders/deconv` and `python -m stune.config`. Use `0.0.0.0:6379` for `STUNE_HOST` and empty for the rest.
1. **In a new terminal**, activate the pcax environment and run the optuna dashboard from the `pcax/examples` folder: `./run_dashboard.sh`. The dashboard is available at `http://127.0.0.1:8080`.
