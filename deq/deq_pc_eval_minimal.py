import time
import os
import sys

import jax
import numpy as np

from deq.deq_pc_minimal_core import eval_on_batch


def evaluate_accuracy(dl, T_steps: int, *, model, optim_h, max_samples: int | None = None) -> float:
    accs = []
    seen = 0
    n_classes = int(model.n_classes.get())
    for x, y in dl:
        y_oh = jax.nn.one_hot(y.numpy(), n_classes)
        accs.append(eval_on_batch(T_steps, x.numpy(), y_oh, model=model, optim_h=optim_h))
        seen += x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    return float(np.mean(accs))


def evaluate_epoch(train_dl, test_dl, T_steps: int, *, model, optim_h, train_eval_samples: int = 10_000):
    t0 = time.perf_counter()
    train_acc = evaluate_accuracy(train_dl, T_steps, model=model, optim_h=optim_h, max_samples=train_eval_samples)
    test_acc = evaluate_accuracy(test_dl, T_steps, model=model, optim_h=optim_h)
    eval_time = time.perf_counter() - t0
    return train_acc, test_acc, eval_time
