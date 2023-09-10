import gc
import logging
import os
import signal
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Optional

from pc_decoder.data_loading import download_datasets
from pc_decoder.params import Params
from pc_decoder.training import TrainingRun, run_training_experiment
from ray import air, tune
from ray.tune.result import DONE
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

# Environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

logging.basicConfig(
    format="%(process)d.%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def signal_handler(signum, frame):
    signame = signal.Signals(signum).name
    stopper = SignalExperimentStopper()
    logging.warning(
        f"Stopper {id(stopper)}: Experiment '{stopper.experiment_name}' received signal {signame} ({signum}), stopping it."
    )
    stopper.stop_experiment()


def register_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)


class SignalExperimentStopper:
    _instance: Optional["SignalExperimentStopper"] = None

    def __new__(cls) -> "SignalExperimentStopper":
        if cls._instance is not None:
            return cls._instance
        return super().__new__(cls)

    def __init__(self):
        if self.__class__._instance is not None:
            return
        register_signal_handlers()
        self.__class__._instance = self
        self.experiment_name = ""
        self._should_stop = False

    def should_stop(self, metrics: dict) -> bool:
        if self._should_stop:
            logging.warning(
                f"Stopper {id(self)} has asked experiment {self.experiment_name} to stop."
            )
        return self._should_stop

    def set_experiment_name(self, name: str) -> None:
        self.experiment_name = name

    def stop_experiment(self) -> None:
        self._should_stop = True
        logging.warning(
            f"Stopper {id(self)} will stop experiment {self.experiment_name} after the current iteration."
        )


def build_trainable(params: Params) -> type[tune.Trainable]:
    class Trainable(tune.Trainable):
        # All overriden methods of this class are called in a Ray worker process.
        def setup(self, config: Dict):
            self.params = params.update(config, inplace=False, validate=True)
            self.params.update(
                {
                    "trial_id": self.trial_id,
                    "trial_name": self.trial_name,
                    "trial_dir": self.logdir,
                },
                inplace=True,
                validate=False,
            )

            # Make sure the datasets are present on this worker
            download_datasets(self.params)

            # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.utils.wait_for_gpu.html#ray.tune.utils.wait_for_gpu
            if self.params.hypertunning_gpu_memory_fraction_per_trial > 0:
                logging.info(f"Run {self.trial_id} is waiting for a GPU...")
                tune.utils.wait_for_gpu(  # type: ignore
                    target_util=1.0
                    - self.params.hypertunning_gpu_memory_fraction_per_trial,
                    retry=50,
                    delay_s=12,
                )
                logging.info(f"Trial {self.trial_id} has enough GPU memory to start.")

            run_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            run_name = f"{self.params.experiment_name}--{self.trial_id}--{run_time}"
            self.training_run = TrainingRun(params=self.params, name=run_name)
            self.stopper = SignalExperimentStopper()
            self.stopper.set_experiment_name(self.training_run.name)

        def step(self) -> dict:
            assert self.training_run is not None
            assert self.stopper is not None
            if self.training_run.epochs_done >= self.params.epochs:
                raise RuntimeError("Number of maximum epochs exceeded!")

            epoch_results = self.training_run.train_for_epoch()
            metrics = epoch_results.report
            if self.stopper.should_stop(metrics):
                logging.warning(
                    f"Experiment stopper decided to stop the experiment after {self.training_run.epochs_done} epochs"
                )
                metrics[DONE] = True

            return metrics

        def save_checkpoint(self, checkpoint_dir: str) -> str | Dict | None:
            assert self.training_run is not None
            self.training_run.save(checkpoint_dir)
            return checkpoint_dir

        def load_checkpoint(self, checkpoint: Dict | str) -> None:
            assert isinstance(checkpoint, str)
            if self.training_run is not None:
                raise RuntimeError(
                    f"Training run {self.training_run.name} already exists and will be overwritten when loading a checkpoint!"
                )
            self.training_run = TrainingRun.load(checkpoint)
            self.params = self.training_run.params.copy()
            self.stopper = SignalExperimentStopper()
            self.stopper.set_experiment_name(self.training_run.name)

        def cleanup(self):
            # This is only called when a Ray worker is cleaned up, not when a trial is cleaned up.
            # Make sure reuse_actors=False is set in the TuneConfig so that the worker is killed at the end of each trial
            # and thus this method is called at the end of each trial.
            logging.info(f"Cleaning up trial {self.trial_id}")
            if self.training_run is not None:
                self.training_run.finish()
                self.training_run = None
                self.stopper = None

    return Trainable


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
        trainable = build_trainable(params)
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
            scheduler = ASHAScheduler(
                time_attr="epochs",
                grace_period=params.hypertunning_early_stop_grace_period_epochs,
            )
        experiment_dir = os.path.join(params.results_dir, params.experiment_name)
        if params.hypertunning_resume_run and tune.Tuner.can_restore(experiment_dir):
            logging.info(
                f"Resuming previous hypertunning run with the same name: {params.experiment_name}"
            )
            logging.warning(
                "Resuming unfinished and errored trials may overwrite their result directories!"
            )
            tuner = tune.Tuner.restore(
                experiment_dir,
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
            )
        else:
            logging.info(f"Starting new hypertunning run: {params.experiment_name}")
            tuner = tune.Tuner(
                trainable,
                param_space=param_space,
                tune_config=tune.TuneConfig(
                    num_samples=params.hypertunning_num_trials,
                    metric="test_mse",
                    mode="min",
                    search_alg=search_algo,
                    scheduler=scheduler,
                    # WARNING: Never reuse actors!
                    # Trials can only be properly finished if each worker executes only one trial.
                    # This is because Ray Tune has no mechanism to notify when a trial is being early stopped by a scheduler.
                    # The only opportunity to summarize and finish an early stopped trial is when cleaning up the worker.
                    # Besides, reusing actors might accumulate RAM usage over time and lead to OOM.
                    reuse_actors=False,
                ),
                run_config=air.RunConfig(
                    stop={"training_iteration": params.epochs},
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
        run_training_experiment(params=params, stopper=SignalExperimentStopper())


if __name__ == "__main__":
    main()
