from typing import Optional

from hyperparameters import HP, Hyperparams  # type: ignore
from hyperparameters.ray_tune_hyperparams import RayTuneHyperparamsMixin  # type: ignore
from ray import tune


class Params(Hyperparams, RayTuneHyperparamsMixin):
    pc_mode: str = HP(
        "Predictive Coding mode: PC or PPC. "
        "PC updates only X each T iteration and then updates W once. "
        "PPC updates both X and W every T iteration. "
        "EfficientPPC first updates X till convergence and then updates W for the rest of the T iterations.",
        default="ppc",
        choices=[
            "pc",
            "ppc",
            "efficient_ppc",
        ],
        tunable=False,
    )
    internal_dim: int = HP(
        "Dimension of the internal representation. Must be internal_dim << output_dim.",
        default=16,
    )
    hidden_dim: int = HP(
        "Dimension of the hidden layers.",
        default=500,
        search_space=tune.choice([500, 750, 1000]),
        tunable=True,
    )
    output_dim: int = HP(
        "Dimension of the data. Must be output_dim >> internal_dim.",
        default=784,
    )
    num_hidden_layers: int = HP(
        "Number of the hidden layers in the generator, excluding input and output layers.",
        default=1,
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
    )

    experiment_name: str = HP(
        "Name of the experiment. An experiment contains many similar runs.",
        default="dev",
    )
    load_weights_from: Optional[str] = HP(
        "Path to the directory containing saved W weights. Note that X values are not loaded (and not saved).",
        default=None,
    )
    epochs: int = HP(
        "Number of epochs to train for.",
        default=250,
    )
    batch_size: int = HP(
        "Number of examples in a batch. Note the last batch will be discarded. Make sure all batches are of the same size!",
        default=500,
    )
    use_last_n_batches_to_compute_metrics: int = HP(
        "Number of last train batches in the epoch used to compute average metrics on the train dataset",
        default=5,
    )
    T: int = HP(
        "Number of Predictive Coding iterations.",
        default=8,
        search_space=tune.choice([4, 8, 12, 20]),
        tunable=True,
    )
    T_max_convergence: int = HP(
        "Maximum number of T iterations that can be performed when optimizing for energy convergence during evaluation. "
        "This parameter is needed to fairly compare models with different T values: even models with very small T values should get a chance to converge during evaluation. "
        "This is different from T, which is used during training.",
        default=100,
    )
    T_min_x_updates: int = HP(
        "For EfficientPPC only. Minimum number of X updates before checking for X convergence.",
        default=1,
    )
    T_min_w_updates: int = HP(
        "For EfficientPPC only. Minimum number of W updates after X updates. The X will be updated till convergence if T budget allows, W will be updated for the rest of the T iterations.",
        default=1,
    )
    energy_quick_approximate_convergence_threshold: float = HP(
        "Upper threshold for the energy convergence. Used when speed is more important than accuracy. If the energy change is less than this threshold, the updates will stop.",
        default=1.0,
    )
    energy_slow_accurate_convergence_threshold: float = HP(
        "Lower threshold for the energy convergence. Used when accuracy is more important than speed. If the energy change is less than this threshold, the updates will stop.",
        default=1e-3,
    )

    optimizer_x: str = HP(
        "Optimizer to use for PC node X values",
        default="sgd",
        choices=["adamw", "sgd"],
        tunable=False,
    )
    optimizer_x_learning_rate: float = HP(
        "Learning rate for PC node values",
        # [5e-2, 1e-1]
        default=5e-2,
        search_space=tune.loguniform(4e-2, 1),
        tunable=True,
    )
    optimizer_x_weight_decay: float = HP(
        "Weight decay for PC node values",
        default=0.0,
    )
    optimizer_x_sgd_momentum: float = HP(
        "Nesterov momentum for SGD optimizer for X",
        default=0.6,
        search_space=tune.loguniform(0.3, 0.7),
        tunable=True,
    )
    optimizer_x_adamw_beta1: float = HP(
        "First momentum parameter for AdamW optimizer for X",
        default=0.9,
        search_space=tune.loguniform(0.5, 0.9999),
        tunable=True,
    )
    optimizer_x_adamw_beta2: float = HP(
        "Second momentum parameter for AdamW optimizer for X",
        default=0.999,
        search_space=tune.loguniform(0.5, 0.9999),
        tunable=True,
    )
    reset_optimizer_x_state: bool = HP(
        "Since we updated the values of x directly, we need to reset the momentums in the x optimizer. Recommended for Adam and AdamW optimizers for X.",
        default=True,
    )

    optimizer_w: str = HP(
        "Optimizer to use for model weights",
        default="sgd",
        choices=["adamw", "sgd"],
        tunable=False,
    )
    optimizer_w_learning_rate: float = HP(
        "Learning rate for model weights",
        default=9.95e-4,
        search_space=tune.loguniform(9e-4, 2e-2),
        tunable=True,
    )
    optimizer_w_weight_decay: float = HP(
        "Weight decay for model weights.",
        default=1e-4,
        search_space=tune.loguniform(1e-5, 1e-3),
        tunable=True,
    )
    optimizer_w_sgd_momentum: float = HP(
        "Nesterov momentum for SGD optimizer for W",
        default=0.9,
        search_space=tune.loguniform(0.7, 0.9999),
        tunable=True,
    )
    optimizer_w_adamw_beta1: float = HP(
        "First momentum parameter for AdamW optimizer for W",
        default=0.9,
        search_space=tune.loguniform(0.5, 0.9999),
        tunable=True,
    )
    optimizer_w_adamw_beta2: float = HP(
        "Second momentum parameter for AdamW optimizer for W",
        default=0.999,
        search_space=tune.loguniform(0.5, 0.9999),
        tunable=True,
    )

    data_dir: str = HP(
        "Directory to save data to.",
        default="data",
        adjust_relative_path=True,
    )
    results_dir: str = HP(
        "Directory to save results to.",
        default="results",
    )
    overwrite_results_dir: bool = HP(
        "Whether to overwrite the results directory",
        default=False,
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
        default=10,
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
    log_t_metrics: bool = HP(
        "Whether to report metrics for each T iteration",
        default=False,
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
        default=300,
    )
    hypertunning_max_concurrency: Optional[int] = HP(
        "Maximum number of concurrent hypertunning trials. "
        "When None, will be determined based on available system resources",
        default=None,
    )
    hypertunning_cpu_per_trial: float = HP(
        "Logical number of CPUs required for a single trial",
        default=1.0,
    )
    hypertunning_ram_gb_per_trial: float = HP(
        "GB of RAM required for a single trial.",
        default=1.0,
    )
    hypertunning_gpu_memory_fraction_per_trial: float = HP(
        "Logical fraction of GPU memory required for a single trial. Must be in [0, 1]. "
        "However, keep in mind that GPUs have only around 85%-90% memory free when sitting idle",
        default=0.025,
    )
    hypertunning_use_early_stop_scheduler: bool = HP(
        "Whether to enable ray.tune scheduler that performs early stopping of trials.",
        default=True,
    )
    hypertunning_early_stop_grace_period_epochs: int = HP(
        "Number of epochs to wait before stopping a trial that is underperforming.",
        default=25,
    )
