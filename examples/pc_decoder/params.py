from ray import tune
from typing import Optional

from hyperparameters import HP, Hyperparams
from hyperparameters.ray_tune_hyperparams import RayTuneHyperparamsMixin


class ModelParams(Hyperparams):
    internal_dim: int = HP(
        "Dimension of the internal representation. Must be internal_dim << output_dim.",
        default=16,
    )
    hidden_dim: int = HP("Dimension of the hidden layers.", default=64)
    output_dim: int = HP(
        "Dimension of the data. Must be output_dim >> internal_dim.", default=784
    )
    num_hidden_layers: int = HP(
        "Number of layers in the generator, including the output layer.", default=2
    )
    activation: str = HP(
        "Activation function to use in the generator.",
        default="relu",
        choices=["gelu", "relu", "tanh", "sigmoid"],
        tunable=True,
    )
    # init_rand_weight: float = HP("Determines the fraction of randomness in the initialization value.", default=0.0)
    # init_forward_weight: float = HP(
    #     "Determines the fraction of forward-pass value in the initialization value.", default=1.0
    # )
    # init_constant: float = HP("A constant to initialize the values.", default=0.0)
    # init_constant_weight: float = HP(
    #     "Determines the fraction of the constant in the initialization value.", default=0.0
    # )


class Params(ModelParams, RayTuneHyperparamsMixin):
    experiment_name: str = HP(
        "Name of the experiment. An experiment contains many similar runs.",
        default="best-parameters-long-run",
    )
    load_weights_from: Optional[str] = HP(
        "Path to the directory containing saved W weights. Note that X values are not loaded (and not saved).",
        default=None,
    )
    epochs: int = HP("Number of epochs to train for.", default=100)
    batch_size: int = HP(
        "Number of examples in a batch. Note the last batch will be discarded. Make sure all batches are of the same size!",
        default=100,
    )
    use_last_n_batches_to_compute_metrics: int = HP(
        "Number of last batches in the epoch used to compute average metrics on the train dataset",
        default=10,
    )
    T: int = HP(
        "Number of Predictive Coding iterations.",
        default=4,
        choices=[1, 4, 8, 16],
        tunable=True,
    )
    optim_x_lr: float = HP(
        "Learning rate for PC node values",
        default=0.02256,
        search_space=tune.loguniform(1e-5, 1e-1),
        tunable=True,
    )
    optim_x_l2: float = HP(
        "Weight decay for PC node values",
        default=0.3935,
        search_space=tune.loguniform(1e-2, 5e-1),
        tunable=True,
    )
    optim_w_lr: float = HP(
        "Learning rate for model weights",
        default=0.0002183,
        search_space=tune.loguniform(1e-5, 1e-1),
        tunable=True,
    )
    optim_w_l2: float = HP(
        "Weight decay for model weights.",
        default=0.17,
        search_space=tune.loguniform(1e-2, 5e-1),
        tunable=True,
    )
    optim_w_momentum: float = HP("Momentum for model weights.", default=0.9)
    optim_w_nesterov: bool = HP(
        "Whether to use Nesterov for model weights", default=True
    )

    data_dir: str = HP(
        "Directory to save data to.", default="data", adjust_relative_path=True
    )
    results_dir: str = HP("Directory to save results to.", default="results")
    overwrite_results_dir: bool = HP(
        "Whether to overwrite the results directory", default=False
    )

    save_best_results: bool = HP(
        "Whether to save the best model",
        default=True,
    )
    save_intermediate_results: bool = HP(
        "Whether to save the intermediate models, graphs, and reports after every N epochs",
        default=False,
    )
    save_results_every_n_epochs: int = HP(
        "Save the intermediate results after every N epochs",
        default=4,
    )
    use_gpus: str = HP(
        "Comma-separated list of GPUs to use",
        default="0",
    )

    do_hypertunning: bool = HP("Whether to do hypertunning.", default=False)
    hypertunning_num_trials: int = HP(
        "Number of hypertunning run trials",
        default=100,
    )
    hypertunning_max_concurrency: Optional[int] = HP(
        "Maximum number of concurrent hypertunning trials. "
        "When None, will be determined based on available system resources",
        default=None,
    )
    hypertunning_cpu_per_trial: float = HP(
        "Logical number of CPUs required for a single trial",
        default=2.0,
    )
    hypertunning_gpu_memory_fraction_per_trial: float = HP(
        "Logical fraction of GPU memory required for a single trial. Must be in [0, 1]. "
        "However, keep in mind that GPUs have only around 85%-90% memory free when sitting idle",
        default=1 / 25,
    )
    hypertunning_use_early_stop_scheduler: bool = HP(
        "Whether to enable ray.tune scheduler that performs early stopping of trials.",
        default=False,
    )
