import time
import numpy as np
import json
import atexit
import os
import sys


class timed_fn:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.times = []
        
        def _save():
        
            _mean = np.mean(self.times[1:])
            _std = np.std(self.times[1:])
            with open(f"{os.path.splitext(os.path.basename(sys.argv[0]))[0]}_clock.json", "w") as f:
                json.dump({
                    "mean": _mean / 1.0e9,
                    "std": _std / 1.0e9
                }, f)
        atexit.register(_save)
    
    def __call__(self, *args, **kwargs):
        _start = time.perf_counter_ns()
        
        r = self.fn(*args, **kwargs)
        
        _end = time.perf_counter_ns()
        self.times.append(_end - _start)
        
        return r
