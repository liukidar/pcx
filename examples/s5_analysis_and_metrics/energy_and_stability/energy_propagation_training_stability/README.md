# Training Stability / Energy Propagation

The code provided here replicates the experiments in Section 5.1: Error propagation and Training Stability.

The two entry points to train models and collect data are:

```
error_propagation.py
training_stability.py
```

Notebook files with the same names, `error_propagation.ipynb` and `training_stability.ipynb`, load saved results and create plots.

## General information

1. `config` folder contains config `yaml` files that are loaded using `hydra-core`.
2. The `artifacts` folder will contain pickle files with training results.
3. Each time a training is started, a random hash is generated has run id. Results are stored to `artifacts/<run-id>`.
4. `logs` will contain the `stdout` and `stderr` when using the docker batch scripts (see below).
5. Plots will be stored to `artifacts/`.
6. `utils/` contains various helper scripts such as data loading, model definitions, custom layers and more.

## How to Run

### Docker

**Single Run**

1. Create a directory `artifacts` with subdirectories `error_propagation`, `training_stability`. Create directories `data` and `logs`.
2. Build the docker using [](/docker/build.sh)
3. Do docker run from this directory. Replace "<this>" with your paths / devices. Here is an example for `training_stability.py`:

```
docker run -it -v $(pwd)/:/home/benchmark -v </host/path/to/pcax>:/home/pcax_tmp -v </path/to/data>:/data -v </path/to/artifacts>:/artifacts -v </path/to/logs:/logs --gpus '"device=<0>"' pcax:latest /bin/bash -c "cd /home/benchmark && ./startup.sh && CLUSTER_NAME=DOCKER python training_stability.py data=fashion_mnist"
```

3. The script uses `hydra-core` to configure run parameters, so overwrite any parameters you want.

**All Runs**

1. To create a batch script for _all_ required runs, use `training_stability_create_launch_scripts.py`, replace the paths therein with yours.
2. Run the newly created `training_stability.sh`.

### Run directly

1. create an environment install `jax` and `pcax` with dependencies. Further, install `seaborn`, `tqdm`, and `hydra-core` (the exact versions etc can be found in this repo in the main directory).
2. add your system to `config/system.yaml`. The config for `YOURCOMPUTER` is a placeholder for you to overwrite. Set an environment variable with `CLUSTER_NAME=<WHATEVER_NAME_YOU_GIVE_YOUR_COMPUTER>`.
3. Create the relevant folders for artifacts, data and logs that you referenced in 2 with subdirectories as outlined above.
4. Run the script directly. For example: `CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_PREALLOCATE=false python training_stability.py data=two_moons`. The script for `error_propagation.py` is equivalent.
