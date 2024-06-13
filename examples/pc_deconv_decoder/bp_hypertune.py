import argparse
from pathlib import Path
import json

import stune
import numpy as np

from bp_deconv import run_experiment


def main(run_info: stune.RunInfo, checkpoint_dir: Path | None = None, seed: int | None = None):
    best_loss = run_experiment(
        dataset_name=run_info["dataset_name"],
        seed=run_info["seed"] if seed is None else seed,
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
    )

    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Configuration file")
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
                results[seed] = float(res)
        print(json.dumps(results, indent=4))
        if results:
            best_seed = min(results, key=results.get)
            v = np.array(list(results.values()))
            top5 = np.sort(v)[:5]
            stats = {
                "best_seed": best_seed,
                "best_loss": results[best_seed],
                "avg5": np.mean(top5),
                "std5": np.std(top5),
                "avg": np.mean(v),
                "std": np.std(v),
            }
            print(json.dumps(stats, indent=4))
        else:
            print("No results")
