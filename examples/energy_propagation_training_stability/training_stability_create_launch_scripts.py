import os
from typing import Any, Dict, List


def list2str(x: List) -> str:
    return "[" + ",".join(map(str, x)) + "]"


def formatting_command(
    run_docker: bool,
    device: int,
    parameters: Dict[str, Any],
    data_path: str,
    internal_log_path: str,
    artifact_path: str,
    pcax_path: str,
    log_path: str = "output.log",
) -> List[str]:
    if run_docker:
        str = "docker run -it "
        str += f"-v $(pwd)/:/home/benchmark -v {pcax_path}:/home/pcax_tmp -v {data_path}:/data -v {artifact_path}:/artifacts -v {internal_log_path}:/logs "
        str += f"--gpus '\"device={device}\"' pcax:latest "
        str += '/bin/bash -c "cd /home/benchmark && ./startup.sh && '
        str += "CLUSTER_NAME=DOCKER "
    else:
        str = f"CUDA_VISIBLE_DEVICES={device} XLA_PYTHON_CLIENT_PREALLOCATE=false "
    str += "python training_stability.py "

    for par_name, par_value in parameters.items():
        if par_value is not None:
            if isinstance(par_value, list):
                par_value = "[" + ",".join(map(str, par_value)) + "]"
            str += f"{par_name}={par_value} "

    if run_docker:
        str += '" '
    str += f"2>&1 | tee {log_path}"

    return str


n_parallel_per_dataset = {
    "two_moons": 16,
    "fashion_mnist": 8,
}

n_seeds_per_dataset = {
    "two_moons": 10,
    "fashion_mnist": 3,
}

w_lr_per_dataset = {
    "two_moons": 0.03,
    "fashion_mnist": 0.01,
}

T_per_method = {"PC": 8, "BP": 1}


h_lr_steps_per_method = {"PC": None, "BP": "[1]"}
h_lr_scalers_per_method = {"PC": None, "BP": "[1]"}

momentums_per_optimizer_w = {"sgd": [0.0, 0.9], "adamw": [0.9]}

hidden_dims_constant_scaling = {
    False: list2str([32, 64, 128, 256, 512, 1024, 2048, 4096]),
    True: list2str([int(14**2), int(18**2), int(23**2), int(28**2), int(33**2), int(38**2), int(42**2), int(48**2)]),
}

datasets_constant_scaling = {False: ["fashion_mnist", "two_moons"], True: ["fashion_mnist"]}


def main():
    conditions = []
    methods = ["BP", "PC"]
    optimizers_w = ["sgd", "adamw"]
    datasets = datasets_constant_scaling[CONSTANT_LAYER_SIZE]

    for dataset in datasets:
        for method in methods:
            for optimizer_w in optimizers_w:
                for momentum in momentums_per_optimizer_w[optimizer_w]:
                    pars = {
                        "data": dataset,
                        "model.definition": method,
                        "optim.w.optimizer": optimizer_w,
                        "optim.w.lr": w_lr_per_dataset[dataset],
                        "run.n_parallel": n_parallel_per_dataset[dataset],
                        "experiment.seeds": n_seeds_per_dataset[dataset],
                        "optim.h.T": T_per_method[method],
                        "experiment.h_lr_steps": h_lr_steps_per_method[method],
                        "experiment.h_lr_scalars": h_lr_steps_per_method[method],
                        "experiment.h_dims": hidden_dims_constant_scaling[CONSTANT_LAYER_SIZE],
                        "optim.w.momentum": momentum,
                    }
                    if CONSTANT_LAYER_SIZE:
                        pars |= {"model.constant_layer_size": True, "data.resize.enabled": True, "run.reload_data": True}
                    conditions.append(pars)

    commands = []
    for i, cond in enumerate(conditions):
        cmd = formatting_command(
            RUN_IN_DOCKER,
            0,
            cond,
            DATA_PATH,
            LOG_PATH,
            ARTIFACT_PATH,
            PCAX_PATH,
            f"logs/output-ts-{cond['data']}-{cond['model.definition']}-{i}.log",
        )
        commands.append(cmd)

    # save to file: training_stability.sh. start with #!/bin/bash
    filename = "training_stability.sh" if not CONSTANT_LAYER_SIZE else "training_stability_ablation.sh"
    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        for cmd in commands:
            f.write(cmd + "\n")

    os.system(f"chmod +x {filename}")

    print("Done")


if __name__ == "__main__":
    # PCAX_PATH = "path/to/pcax_no_install"
    RUN_IN_DOCKER = False
    CONSTANT_LAYER_SIZE = True
    PCAX_PATH = "/home/cornelius/Projects/pcax_no_install"
    DATA_PATH = "/mnt/large/data"
    LOG_PATH = "/mnt/large/logs/PC-Benchmark"
    ARTIFACT_PATH = "/mnt/large/artifacts/PC-Benchmark"
    main()
