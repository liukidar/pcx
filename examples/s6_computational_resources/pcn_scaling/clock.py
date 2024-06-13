import time
import numpy as np
import json
import os
import sys
import jax


class timed_fn:
    def __init__(self, fn, params) -> None:
        self.fn = fn
        self.times = []
        self.params = params
        
    def save(self):        
        return {"seconds": self.times[1:], "params": self.params}
    
    def __call__(self, *args, **kwargs):
        _start = time.perf_counter_ns()
        
        r = self.fn(*args, **kwargs)
        jax.block_until_ready(r)
        
        _end = time.perf_counter_ns()
        self.times.append((_end - _start) / 1e9)
        
        return r
