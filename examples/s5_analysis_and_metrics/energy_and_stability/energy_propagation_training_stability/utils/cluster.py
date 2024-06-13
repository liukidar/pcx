from typing import Any, Dict
import yaml
import sys
import os


class DBConnect:
    def __init__(self, host: str, port: int, user: str, password: str, name: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.name = name


class ClusterManager:
    def __init__(self, name: str = None, auto: bool = True):
        """Creates a ClusterManager object that can automatically configure multiple clusters.

        Args:
            name (str): Name of the cluster as given in YAML.
            auto (bool, optional): Whether the cluster should be identified automatically from the linux environment variables. Defaults to True.

        Raises:
            OSError: YAML config file not found.
            NotImplementedError: Cluster ID not found in YAML config file.
        """
        if auto:
            sys_name = os.getenv("CLUSTER_NAME")
            if sys_name is None:
                raise OSError("CLUSTER_NAME not found in environment variables. Autoselecting system failed.")
            else:
                self.name = sys_name
        else:
            self.name = name

        config_file = "config/system.yaml"
        with open(config_file) as file:
            self._configs = yaml.load(file, Loader=yaml.FullLoader)

        if self.name not in self._configs.keys():
            raise NotImplementedError(f"System {self.name} not implemented in '{config_file}'")

        self._configs = self._configs[self.name]

    @property
    def project_dir(self) -> str:
        return self._configs["PROJECT_DIR"]

    @property
    def num_workers(self) -> int:
        return self._configs["NUM_WORKERS"]

    @property
    def data_dir(self) -> str:
        return self._configs["DATA_DIR"]

    @property
    def log_dir(self) -> str:
        return self._configs["LOG_DIR"]

    @property
    def artifact_dir(self) -> str:
        return self._configs["ARTIFACTS_DIR"]

    @property
    def network(self):
        return self._configs["NETWORK"]

    @property
    def use_GPU(self) -> bool:
        return self._configs["USE_GPU"]

    @property
    def get_pid(self) -> int:
        try:
            return os.environ["SLURM_JOB_ID"]
        except KeyError:
            return os.getpid()

    @property
    def db(self) -> DBConnect:
        return DBConnect(
            host=self._configs["DB_HOST"],
            port=self._configs["DB_PORT"],
            user=self._configs["DB_USER"],
            password=self._configs["DB_PASSWORD"],
            name=self._configs["DB_NAME"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k.lower(): v for k, v in self._configs.items()}
