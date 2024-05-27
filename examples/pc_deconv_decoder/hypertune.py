import argparse
from pathlib import Path

import stune

from pc_deconv import run_experiment


def main(run_info: stune.RunInfo):
    best_loss = run_experiment(
        num_layers=run_info["hp/num_layers"],
        internal_state_dim=(
            run_info["hp/internal_state_dim"],
            run_info["hp/internal_state_dim"],
            run_info["hp/internal_state_channels"],
        ),
        kernel_size=run_info["hp/kernel_size"],
        batch_size=run_info["hp/batch_size"],
        epochs=run_info["hp/epochs"],
        T=run_info["hp/T"],
        optim_x_lr=run_info["hp/optim/x/lr"],
        optim_x_momentum=run_info["hp/optim/x/momentum"],
        optim_w_lr=run_info["hp/optim/w/lr"],
        optim_w_b1=run_info["hp/optim/w/b1"],
        optim_w_b2=run_info["hp/optim/w/b2"],
    )

    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hypertune.yaml", nargs="?", type=str, help="Configuration file")
    parser.add_argument("--checkpoint_dir", default="results/hp", type=Path, help="Directory to save checkpoints")
    args = parser.parse_args()

    main(stune.RunInfo(stune.load_config(args.config)))
