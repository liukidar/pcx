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
        s = "docker run -it "
        s += f"-v $(pwd)/:/home/benchmark -v {pcax_path}:/home/pcax_tmp -v {data_path}:/data -v {artifact_path}:/artifacts -v {internal_log_path}:/logs "
        s += f"--gpus '\"device={device}\"' pcax:latest "
        s += '/bin/bash -c "cd /home/benchmark && ./startup.sh && '
        s += "CLUSTER_NAME=DOCKER "
    else:
        s = f"CUDA_VISIBLE_DEVICES={device} XLA_PYTHON_CLIENT_PREALLOCATE=false "
    s += "python error_propagation.py "

    for par_name, par_value in parameters.items():
        if par_value is not None:
            if isinstance(par_value, list):
                par_value = "[" + ",".join(map(str, par_value)) + "]"
            s += f"{par_name}={par_value} "

    if run_docker:
        s += '" '
    s += f"2>&1 | tee {log_path}"

    return s


n_parallel_per_dataset = {
    "two_moons": 16,
    "two_circles": 16,
    "fashion_mnist": 8,
}

n_seeds_per_dataset = {
    "two_moons": 10,
    "two_circles": 10,
    "fashion_mnist": 3,
}

w_lr_per_dataset = {
    "two_moons": list2str([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-3, 1.0]),
    "two_circles": list2str([1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-3, 1.0]),
    "fashion_mnist": list2str([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]),
}

T_per_method = {"PC": 8, "BP": 1}


h_lr_per_method = {"PC": None, "BP": "[1]"}

momentums_per_optimizer_w = {"sgd": [0.0, 0.5, 0.9, 0.95], "adamw": [0.9]}

hidden_dims_per_dataset = {
    "two_moons": list2str([128, 256, 512, 1024]),
    "two_circles": list2str([128, 256, 512, 1024]),
    "fashion_mnist": list2str([512, 1024, 2048, 4096]),
}


def main():
    conditions = []
    datasets = ["fashion_mnist", "two_moons", "two_circles"]
    methods = ["BP", "PC"]
    optimizers_w = ["sgd", "adamw"]

    for dataset in datasets:
        for method in methods:
            for optimizer_w in optimizers_w:
                for momentum in momentums_per_optimizer_w[optimizer_w]:
                    pars = {
                        "data": dataset,
                        "model.definition": method,
                        "experiment.optimizer_w": [optimizer_w],
                        "experiment.w_lr": w_lr_per_dataset[dataset],
                        "run.n_parallel": n_parallel_per_dataset[dataset],
                        "experiment.seeds": n_seeds_per_dataset[dataset],
                        "optim.h.T": T_per_method[method],
                        "experiment.h_lr": h_lr_per_method[method],
                        "experiment.h_dims": hidden_dims_per_dataset[dataset],
                        "experiment.momentum_w": [momentum],
                    }
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
            f"logs/output-ep-{cond['data']}-{cond['model.definition']}-{i}.log",
        )
        commands.append(cmd)

    # save to file: error_propagation.sh. start with #!/bin/bash

    with open("error_propagation.sh", "w") as f:
        f.write("#!/bin/bash\n")
        for cmd in commands:
            f.write(cmd + "\n")

    os.system("chmod +x error_propagation.sh")

    print("Done")


if __name__ == "__main__":
    RUN_IN_DOCKER = False
    PCAX_PATH = "path/to/pcx"
    DATA_PATH = "/path/to/data"
    LOG_PATH = "/path/to/logs"
    ARTIFACT_PATH = "/path/to/artifacts"
    main()
