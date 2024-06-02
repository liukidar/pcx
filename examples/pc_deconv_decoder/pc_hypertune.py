import argparse
from pathlib import Path

import stune

from pc_deconv import run_experiment


def main(run_info: stune.RunInfo):
    best_loss = run_experiment(
        dataset_name=run_info["dataset_name"],
        num_layers=run_info["hp/num_layers"],
        internal_state_dim=(
            run_info["hp/internal_state_channels"],
            run_info["hp/internal_state_dim"],
            run_info["hp/internal_state_dim"],
        ),
        kernel_size=run_info["hp/kernel_size"],
        act_fn=run_info["hp/act_fn"],
        output_act_fn=run_info["hp/output_act_fn"],
        batch_size=run_info["hp/batch_size"],
        epochs=run_info["hp/epochs"],
        T=run_info["hp/T"],
        optim_x_lr=run_info["hp/optim/x/lr"],
        optim_x_momentum=run_info["hp/optim/x/momentum"],
        optim_w_name=run_info["hp/optim/w/name"],
        optim_w_lr=run_info["hp/optim/w/lr"],
        optim_w_wd=run_info["hp/optim/w/wd"],
        optim_w_momentum=run_info["hp/optim/w/momentum"],
    )

    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pc_hypertune.yaml", nargs="?", type=str, help="Configuration file")
    parser.add_argument("--checkpoint_dir", default="results/pc_hp", type=Path, help="Directory to save checkpoints")
    args = parser.parse_args()

    main(stune.RunInfo(stune.load_config(args.config)))
