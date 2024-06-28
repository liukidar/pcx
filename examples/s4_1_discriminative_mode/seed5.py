import torch
import numpy
import pcax
import random

import os
import sys
import json

from typing import Any, Optional, List

import omegaconf
from omegaconf import OmegaConf

SEEDS = [0, 1, 2, 3, 4, 5, 6]

class RunInfo:
    def __init__(
        self,
        config: OmegaConf,
        study_name: str = None,
        log=None,
    ) -> None:
        self.config = config
        self.study_name = study_name
        self.log = log or {}
        self.locked = False

        OmegaConf.register_new_resolver(
            "py", lambda code: eval(code.strip()), replace=True
        )
        OmegaConf.register_new_resolver(
            "hp", lambda param: self[f"hp/{param}"], replace=True
        )

    def __getitem__(self, i: Any) -> Any:
        if i in self.log:
            return self.log[i]

        if self.locked is True:
            raise PermissionError("Cannot access new parameter from a locked RunInfo")

        path = i.split("/")
        param = self.config
        for key in path:
            param = param[key]

        if isinstance(param, omegaconf.DictConfig):
            param = param.get("default", None)

        self.log[i] = param

        return param

    def __setitem__(self, i: Any, v: Any) -> None:
        if self.locked is True:
            raise PermissionError("Cannot modify parameter in a locked RunInfo")

        path = i.split("/")
        param = self.config
        for key in path[:-1]:
            param = param[key]
        param[path[-1]] = v

        self.log[i] = v

    def lock(self, to_load: List[str] = []):
        # Load required elements before locking
        for p in to_load:
            self.__getitem__(p)

        self.locked = True


class run:
    def __init__(self, fn):
        self._seeds = SEEDS

        def wrap_fn(*args, **kwargs):
            best_per_seed = []
            accuracies_per_seed = []
            betst_per_seed5 = []
            accuracies_per_seed5 = []
            for seed in self._seeds:
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)
                pcax.RKG.seed(seed)
                
                best, best5, accuracies, accuracies5 = fn(*args, **kwargs)
                best_per_seed.append(best)
                accuracies_per_seed.append(accuracies)
                betst_per_seed5.append(best5)
                accuracies_per_seed5.append(accuracies5)
            
            return best_per_seed, accuracies_per_seed, betst_per_seed5, accuracies_per_seed5
        self._fn = wrap_fn
        
    def __call__(self, *args, **kwargs):
        best, accuracies, best5, accuracies5 = self._fn(*args, **kwargs)
        
        with open(
            f"{sys.argv[1]}_accuracy.json",
            "w"
        ) as f:
            top5 = numpy.sort(best)[2:]
            top5_ = numpy.sort(best5)[2:]
            json.dump({
                "accuracies": accuracies,
                "avg": numpy.mean(best),
                "std": numpy.std(best),
                "avg5": numpy.mean(top5),
                "std5": numpy.std(top5),
                "accuracies_5": accuracies5,
                "avg_5": numpy.mean(best5),
                "std_5": numpy.std(best5),
                "avg5_5": numpy.mean(top5_),
                "std5_5": numpy.std(top5_)

            }, f, indent=4)