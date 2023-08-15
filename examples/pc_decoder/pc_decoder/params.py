from typing import Optional

from hyperparameters import HP, Hyperparams
from hyperparameters.ray_tune_hyperparams import RayTuneHyperparamsMixin
from ray import tune


class ModelParams(Hyperparams):
    pc_mode: str = HP(
        "Predictive Coding mode: PC or PPC. "
        "PC updates only X each T iteration and then updates W once. "
        "PPC updates both X and W every T iteration.",
        default="pc",
        choices=["pc", "ppc"],
        tunable=True,
    )
    internal_dim: int = HP(
        "Dimension of the internal representation. Must be internal_dim << output_dim.",
        default=16,
    )
    hidden_dim: int = HP(
        "Dimension of the hidden layers.",
        default=256,
        search_space=tune.choice([32, 64, 128, 256, 512, 1024]),
        tunable=True,
    )
    output_dim: int = HP(
        "Dimension of the data. Must be output_dim >> internal_dim.", default=784
    )
    num_hidden_layers: int = HP(
        "Number of the hidden layers in the generator, excluding input and output layers.",
        default=1,
        search_space=tune.choice([1, 2, 3, 4]),
        tunable=True,
    )
    activation_hidden: str = HP(
        "Activation function to use in the generator.",
        default="gelu",
    )
    activation_output: Optional[str] = HP(
        "Activation function to use in the output layer of the generator. None means no activation function on the output layer.",
    )
    use_prior_layer: bool = HP(
        "Whether to use a prior layer that generates a set of priors for the topmost PCLayer.",
        default=False,
        tunable=True,
    )
    T_convergence_multiplier: int = HP(
        "Multiplier for the number of iterations to achieve approximate convergence.",
        default=1,
    )
    preserve_internal_state_between_batches: bool = HP(
        "Whether to preserve the x values of the first PCLayer in between batches during training",
        default=True,
    )
    preserve_all_pc_states_between_batches: bool = HP(
        "Whether to preserve the x values of all PCLayers in between batches during training",
        default=False,
    )


class Params(ModelParams, RayTuneHyperparamsMixin):
    experiment_name: str = HP(
        "Name of the experiment. An experiment contains many similar runs.",
        default="dev",
    )
    load_weights_from: Optional[str] = HP(
        "Path to the directory containing saved W weights. Note that X values are not loaded (and not saved).",
        default=None,
    )
    epochs: int = HP("Number of epochs to train for.", default=10)
    batch_size: int = HP(
        "Number of examples in a batch. Note the last batch will be discarded. Make sure all batches are of the same size!",
        default=1,
    )
    use_last_n_batches_to_compute_metrics: int = HP(
        "Number of last train batches in the epoch used to compute average metrics on the train dataset",
        default=1,
    )
    T: int = HP(
        "Number of Predictive Coding iterations.",
        default=100,
        search_space=tune.choice(list(range(8, 100, 3))),
        tunable=True,
    )
    optim_x_lr: float = HP(
        "Learning rate for PC node values",
        # default=0.357,
        default=0.2,
        search_space=tune.loguniform(1e-2, 1e2),
        tunable=True,
    )
    optim_x_l2: float = HP(
        "Weight decay for PC node values",
        default=0.0,
        search_space=tune.loguniform(1e-2, 5e-1),
        tunable=False,
    )
    optim_w_lr: float = HP(
        "Learning rate for model weights",
        # default=5.185e-4,
        default=5e-4,
        search_space=tune.loguniform(1e-5, 1e-3),
        tunable=True,
    )
    optim_w_l2: float = HP(
        "Weight decay for model weights.",
        default=0.001,
        search_space=tune.loguniform(1e-2, 5e-1),
        tunable=False,
    )
    optimizer_x: str = HP(
        "Optimizer to use for PC node X values",
        default="adamw",
        choices=["adamw", "sgd"],
    )
    reset_optimizer_x_state: bool = HP(
        "Since we updated the values of x directly, we need to reset the momentums in the x optimizer. Recommended for Adam and AdamW optimizers for X.",
        default=True,
    )
    optimizer_w: str = HP(
        "Optimizer to use for model weights",
        default="adamw",
        choices=["adamw", "sgd"],
    )

    data_dir: str = HP(
        "Directory to save data to.", default="data", adjust_relative_path=True
    )
    results_dir: str = HP("Directory to save results to.", default="results")
    overwrite_results_dir: bool = HP(
        "Whether to overwrite the results directory", default=True
    )
    save_best_results: bool = HP(
        "Whether to save the best model",
        default=True,
    )
    save_intermediate_results: bool = HP(
        "Whether to save the intermediate models, graphs, and reports after every N epochs",
        default=True,
    )
    save_results_every_n_epochs: int = HP(
        "Save the intermediate results after every N epochs",
        default=1,
    )
    visualize_n_images_per_label: int = HP(
        "When visualizing generated images, up to this number of examples per label will be generated.",
        default=2,
    )

    use_gpus: str = HP(
        "Comma-separated list of GPUs to use",
        default="0",
    )

    wandb_logging: bool = HP(
        "Whether to log results to wandb.ai",
        default=True,
    )
    do_hypertunning: bool = HP(
        "Whether to do hypertunning.",
        default=False,
    )
    hypertunning_resume_run: bool = HP(
        "Whether to resume the previous hypertunning run with the same name, if present.",
        default=False,
    )
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
    hypertunning_ram_gb_per_trial: float = HP(
        "GB of RAM required for a single trial.",
        default=4.0,
    )
    hypertunning_gpu_memory_fraction_per_trial: float = HP(
        "Logical fraction of GPU memory required for a single trial. Must be in [0, 1]. "
        "However, keep in mind that GPUs have only around 85%-90% memory free when sitting idle",
        default=1 / 10,
    )
    hypertunning_use_early_stop_scheduler: bool = HP(
        "Whether to enable ray.tune scheduler that performs early stopping of trials.",
        default=False,
    )
