import argparse
from pathlib import Path

import stune
import numpy as np

from bp_deconv import run_experiment


def main(run_info: stune.RunInfo, checkpoint_dir: Path | None = None, seed: int | None = None):
    best_loss = run_experiment(
        dataset_name=run_info["dataset_name"],
        kernel_size=run_info["hp/kernel_size"],
        act_fn=run_info["hp/act_fn"],
        output_act_fn=run_info["hp/output_act_fn"],
        batch_size=run_info["hp/batch_size"],
        epochs=run_info["hp/epochs"],
        optim_w_name=run_info["hp/optim/w/name"],
        optim_w_lr=run_info["hp/optim/w/lr"],
        optim_w_wd=run_info["hp/optim/w/wd"],
        optim_w_momentum=run_info["hp/optim/w/momentum"],
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bp_hypertune.yaml", nargs="?", type=str, help="Configuration file")
    parser.add_argument("--checkpoint_dir", default=None, type=Path, help="Directory to save checkpoints")
    parser.add_argument("--test-seed", default=False, action="store_true", help="Test random seed")
    args = parser.parse_args()

    config = stune.RunInfo(stune.load_config(args.config))

    if not args.test_seed:
        main(config, args.checkpoint_dir)
    else:
        results = {}
        for seed in range(7):
            print(f"Running for seed {seed}")
            res = main(config, args.checkpoint_dir, seed)
            if not np.isnan(res):
                results[seed] = res
        print(results)
        if results:
            best_seed = min(results, key=results.get)
            print(f"Best seed {best_seed}: {results[best_seed]}")
            print(f"Average loss: {np.mean(list(results.values()))}")
