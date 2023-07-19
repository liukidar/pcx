__all__ = ['save_params', 'load_params', 'TorchDataloader']


import numpy as np
import torch.utils
import jax


########################################################################################################################
#
# PARAMETERS LOADING
#
########################################################################################################################


def save_params(params, path):
    keys, values = zip(*jax.tree_util.tree_flatten_with_path(params)[0])
    np.savez_compressed(path, **dict(zip(map(repr, keys), values)))


def load_params(params, path):
    raw_data = np.load(path)
    data, treedef = jax.tree_util.tree_flatten_with_path(params)

    assert all((rd == repr(d[0])) for rd, d in zip(raw_data.keys(), data)), "Parameter names mismatch."

    return jax.tree_util.tree_unflatten(treedef, raw_data.values())


########################################################################################################################
#
# DATA LOADING
#
########################################################################################################################


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )


# BATCH ALIGNED SAMPLER

class BatchAlignedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create data buckets
        buckets = {}
        for i, (x, y) in enumerate(dataset):
            buckets.setdefault(y, []).append(i)

        self.indices_by_class = tuple(
            np.array(bucket) for bucket in buckets.values()
        )
        self.index_len = self.batch_size // len(self.indices_by_class)

        # Assert that batch_size is a multiple of the number of classes
        assert self.batch_size % len(self.indices_by_class) == 0, \
            f"Batch size (b={self.batch_size}) must be a multiple" \
            f" of the number of classes (c={len(self.indices_by_class)})."

    def __iter__(self):
        # Shuffle indices
        if self.shuffle:
            for indices in self.indices_by_class:
                np.random.shuffle(indices)

        yield from np.concatenate(
            tuple(
                np.reshape(indices[:(len(indices) // self.index_len) * self.index_len], (-1, self.index_len))
                for indices in self.indices_by_class
            ),
            axis=1
        )
