__all__ = ["load_params", "save_params"]

from collections.abc import Callable
from types import UnionType
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ..core._parameter import BaseParam, get
from ..core._tree import _cache
from ..nn._parameter import LayerParam

########################################################################################################################
#
# SERIALIZATION
#
# Utilities to save/load a model.
#
########################################################################################################################


def save_params(
    model: PyTree,
    path: str,
    filter: Callable[[Any], bool] | type[BaseParam] = LayerParam,
) -> None:
    """Function to save the parameters of a model to a file. The '.npz' extension is automatically added
    to the file name.

    Args:
        model (PyTree): the model to dump to disk.
        path (str): the path to the file where to save the model. If the file already exists, it will be
            overwritten.
        filter (Callable[[Any], bool] | Type[BaseParam], optional): filter function or type identifying
            the parameters to save. The default value 'LayerParam' selects all the weights of the layers in the
            model.
    """
    _filter_fn = filter if not isinstance(filter, type | UnionType) else lambda x: isinstance(x, filter)

    _params = jtu.tree_flatten_with_path(model, is_leaf=_filter_fn)[0]

    # Cache to check for duplicate parameters: a shared parameter is saved only under its first path.
    _seen = _cache()
    _data = {}
    for key, param in _params:
        if _filter_fn(param):
            assert isinstance(param, BaseParam), "Only parameters can be serialized."
            if _seen(id(param)) is None:
                _data[jtu.keystr(key)] = param.get()

    np.savez_compressed(path, **_data)


def load_params(
    model: PyTree,
    path: str,
    filter: Callable[[Any], bool] | type[BaseParam] = LayerParam,
) -> None:
    """Function to load the parameters of a model from a file. The '.npz' extension is automatically added
    to the file name. The model must have the same structure as the one used to save the parameters and must
    already be initialized:

    ```python
    model = Model()
    load_params(model, "model.npz")
    ```

    Args:
        model (PyTree): target model.
        path (str): the path to the file containing the model parameters to load.
        filter (Callable[[Any], bool] | Type[BaseParam], optional): filter function or type identifying
            the parameters to save. The default value 'LayerParam' selects all the weights of the layers in
            the model.

    Raises:
        KeyError: the file does not contain all the parameters required by the model.
        ValueError: a stored parameter does not have the shape of its target parameter.
    """
    path = path if path.endswith(".npz") else f"{path}.npz"
    _filter_fn = filter if not isinstance(filter, type | UnionType) else lambda x: isinstance(x, filter)

    _loaded_values = np.load(path)
    _params = jtu.tree_flatten_with_path(model, is_leaf=_filter_fn)[0]

    # Mirrors 'save_params': a shared parameter is stored only under its first path.
    _seen = _cache()
    for _key, _param in _params:
        if _filter_fn(_param) and _seen(id(_param)) is None:
            _key = jtu.keystr(_key)
            if _key not in _loaded_values:
                raise KeyError(f"Parameter '{_key}' not found in the file '{path}'.")
            _value = _loaded_values[_key]
            # A target parameter that holds nothing yet (a freshly built Vode, for example) has no shape to
            # compare against, so it accepts whatever the file stores.
            _current = get(_param)
            if _current is not None and jax.numpy.shape(_current) != _value.shape:
                raise ValueError(
                    f"Parameter '{_key}' has shape {jax.numpy.shape(_current)} but the file stores {_value.shape}."
                )
            _param.set(jax.numpy.array(_value))

    _loaded_values.close()
