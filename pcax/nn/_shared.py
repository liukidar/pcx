__all__ = ['shared']

from typing import Type, Callable, Any
from jaxtyping import PyTree
from types import UnionType

import jax.tree_util as jtu

from ..core._parameter import BaseParam
from ..core._tree import tree_ref, tree_unref


####################################################################################################
#
# SHARED
#
# Utility to simplify parameter sharing between modules.
#
####################################################################################################


def shared(module: PyTree, filter: Callable[[Any], bool] | Type[BaseParam] = BaseParam) -> PyTree:
    """Creates a copy of the input pytree which shares all the target parameters with the original.
    It can be used to create modules with weight sharing:
    
    ```python
    linear1 = Linear(10, 10)
    linear2 = shared(linear1)
    ```
    
    The same can be achieved manually if only a subset of the parameters is to be shared:
    
    ```python
    linear1 = Linear(10, 10)
    linear2 = Linear(10, 10)
    
    linear2.nn.weight = linear1.nn.weight
    ```
    
    NOTE #1: jax works exclusively with pytrees, so duplicate references to the same object within
    a pytree are not allowed. pcax partially solves the problem by allowing multiple references to the
    same parameter. Thus, it is not possible to share a whole module, but only to create a new one and
    share the required parameters to it. The following is not allowed:
    
    ```python
    linear1 = Linear(10, 10)
    linear2 = linear1  # WRONG if used in a pytree.
    ```
    
    NOTE #2: flatten/unflatten creates a copy of the tree which, however, references the
    same leaves. By changing what is considered a leaf, we can control what is
    copied and what is shared:
    
    - if 'filter = BaseParam', all parameters are shared.
    - if 'filter = DynamicParam', only dynamic parameters are shared, static parameters are flattened
      and thus copied.

    Args:
        module (PyTree): input pytree to copy.
        filter (Callable[[Any], bool] | Type[BaseParam], optional): filter function or type identifying
            the parameters to share.

    Returns:
        PyTree: copy of the input pytree with shared parameters.
    """
        
    # we ref the tree to preserve its structure even in the new copy (if any parameter is copied and
    # not shared, any reference to it would get duplicated in the new tree).
    _tree, _structure = jtu.tree_flatten(
        tree_ref(module),
        is_leaf=filter 
            if not isinstance(filter, type | UnionType)
            else lambda x: isinstance(x, filter)
    )
    
    return tree_unref(jtu.tree_unflatten(_structure, _tree))
