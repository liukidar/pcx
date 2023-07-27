import logging
import os
from argparse import ArgumentParser

from pc_decoder.data_loading import download_datasets
from pc_decoder.params import Params
from pc_decoder.training import Trainable, run_training_experiment
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

# Environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    parser = ArgumentParser()
    Params.add_arguments(parser)
    args = parser.parse_args()
    params = Params.from_arguments(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = params.use_gpus

    # Download datasets
    download_datasets(params)

    if (
        params.hypertunning_gpu_memory_fraction_per_trial < 0
        or params.hypertunning_gpu_memory_fraction_per_trial > 1
    ):
        raise ValueError(
            f"--hypertunning-gpu-memory-fraction-per-trial must be in [0, 1]"
        )

    if params.do_hypertunning:
        trainable = Trainable(params)
        trainable = tune.with_resources(
            trainable,
            {
                "cpu": params.hypertunning_cpu_per_trial,
                "memory": params.hypertunning_ram_gb_per_trial * 1024**3,
                "gpu": params.hypertunning_gpu_memory_fraction_per_trial,
            },
        )
        param_space = params.ray_tune_param_space()
        search_algo = OptunaSearch(points_to_evaluate=[params.ray_tune_best_values()])
        if params.hypertunning_max_concurrency is not None:
            search_algo = ConcurrencyLimiter(
                search_algo, max_concurrent=params.hypertunning_max_concurrency
            )
        scheduler = None
        if params.hypertunning_use_early_stop_scheduler:
            scheduler = ASHAScheduler()
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                num_samples=params.hypertunning_num_trials,
                metric="test_mse",
                mode="min",
                search_alg=search_algo,
                scheduler=scheduler,
                # This may help avoid XLA compilations that take more than 10 minutes for each new process
                reuse_actors=True,
            ),
            run_config=air.RunConfig(
                name=params.experiment_name,
                local_dir=params.results_dir,
                failure_config=air.FailureConfig(max_failures=3),
            ),
        )
        results = tuner.fit()
        logging.info("--- Hypertunning done! ---")

        # FIXME: use the best reported score to compare results!
        # Note: ray.tune interprets the last reported result of each trial as the "best" one:
        # https://github.com/ray-project/ray_lightning/issues/81
        results.get_dataframe().to_csv(
            os.path.join(
                params.results_dir, params.experiment_name, "hypertunning_results.csv"
            ),
            index=False,
        )
    else:
        run_training_experiment(params)


if __name__ == "__main__":
    main()
