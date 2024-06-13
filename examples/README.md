# Examples

- PCAX comes with examples that showcase how to implement different architectures using the library.
- Examples also include hypertuning code, search spaces, and the best parameters, which makes it easy to experiment and reproduce the results from the paper.
- Both PC and BP versions of architectures are provided for performance and implementation comparison.

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

## Autoencoders: Image Generation

An Autoencoder is a model that learns efficient compression of the input data. A BP-based Autoencoder consists of an encoder, that compresses input data into a much smaller bottleneck dimension, and a decoder, that restores an image from the compressed representation. A PC-based autoencoder consists only of the decoder part. Refer to the `C.1 Autoencoder` appendix of the `PCX: Benchmarking Predictive Coding Networks – Made Simple` paper for more details.

Available Autoencoder models and datasets in the `examples/autoencoders` folder:

| Autoencoder layers | Dataset | Experiment subfolder |
|-------|---------|---------|
| Linear | MNIST | `examples/autoencoders/linear` |
| Linear | FashionMNIST | `examples/autoencoders/linear` |
| Deconvolution (transposed 2D convolution) | CIFAR10 | `examples/autoencoders/deconv` |
| Deconvolution (transposed 2D convolution) | CelebA | `examples/autoencoders/deconv`|

In general:

1. The `configs` folder in each experiment subfolder contains the definitions of hyperparameters, search spaces, and the best values on all datasets. By using different config files with the `--config` parameter you can run different experiments. Check out all the config files to see what is available.
1. If you want to run a single training with the best hyperparameters, use `pc_deconv.py` or `pc_decoder.py` for PC and iPC experiments, and `bp_deconv.py` or `bp_decoder.py` for BP experiments, respecitvely.
1. If you want to test training with different random seeds, use `pc_hypertune.py` for PC and iPC, and `bp_hypertune.py` for BP with the `--test-seed` argument.
1. If you want to run hypertuning, look at the `run_pc_hypertune.sh` and `run_bp_hypertune.sh` scripts. They configure the number of trials and the number of GPUs used and call `pc_hypertune.py` for PC and iPC, and `bp_hypertune.py` for BP, respectively.

You can run both Linear-based models on MNIST and FashionMNIST and Deconv-based models on CIFAR10 and CelebA in exactly the same way. Thus, we will only demonstrate the process using the Deconv-based models on CIFAR10. The main difference is that for the linear models you should use `pc_decoder.py` instead of `pc_deconv.py` and `bp_decoder.py` instead of `bp_deconv.py`.


To run the training with the best parameters:

1. Activate the virtual environment: `conda activate pcax`.
1. `cd examples/autoencoders/deconv`
1. Run PC on CIFAR10: `python pc_deconv.py --config configs/pc_cifar10_adamw_hypertune.yaml`
1. Run iPC on CIFAR10: `python pc_deconv.py --config configs/pc_ipc_cifar10_adamw_hypertune.yaml`
1. Run BP on CIFAR10: `python bp_deconv.py --config configs/bp_cifar10_adamw_hypertune.yaml`

You may notice the results are better than the ones reported in the paper. This is because in the paper we reported results averaged over the 5 best random seeds. You can reproduce the exact results from the paper by running the random seed test.

To run the random seed test use the `--test-seed` parameter:
1. Activate the virtual environment: `conda activate pcax`.
1. `cd examples/autoencoders/deconv`
1. Run PC on CIFAR10: `python pc_hypertune.py --config configs/pc_cifar10_adamw_hypertune.yaml --test-seed`
1. Run iPC on CIFAR10: `python pc_hypertune.py --config configs/pc_ipc_cifar10_adamw_hypertune.yaml --test-seed`
1. Run BP on CIFAR10: `python bp_hypertune.py --config configs/bp_cifar10_adamw_hypertune.yaml --test-seed`

When you change the model implementation or the hyperparameter search space you should run a full hypertuning. First make sure you have followed the stune configuration steps outlined above.

To run hypertuning for PC or iPC (similar for BP, just use the `run_bp_hypertune.sh` script):
1. Edit `run_pc_hypertune.sh` to configure the GPUs to use and the number of trials. The total number of trials is `num_gpus * n_trials / gpus_per_task`, where `n_trials` is the number before `:` in the `--n_trails 32:8`, and `gpus_per_task` is specified in the config file.
1. `cd pcax/examples/autoencoders/deconv` where the `.stune` folder is located. If the `.stune` folder is not present, you have probably ommitted the stune configuration in this folder. Go back to the stune configuration instructions.
1. Run the hypertuning: `./run_pc_hypertune.sh cifar10_adamw`
1. Note that you should specify the name of the config file from the `configs` directory as a parameter to the `run_pc_hypertune.sh` script, but without the `pc_` prefix and the `_hypertune.yaml` postfix. The same is for BP, but without the `bp_` prefix.
1. You can observe logs in the `.stune/output` folder.
1. You can view the optuna dashboard at `http://127.0.0.1:8080`.

