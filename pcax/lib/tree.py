import jax.tree_util as jtu
from typing import Any, Callable
import copy


# Function used to create a mask of a pytree containing
# the names of each node.
# It is useful to create a dictionary-like external state.
# This is currently not necessary as the state is stored
# within the model.
def _get_named_leaves(node, leaves):
    keys = None
    try:
        dict = vars(node)
        keys = list(map(lambda el: "." + el, dict.keys()))
        elements = list(dict.values())
    except Exception:
        pass

    if keys is None:
        try:
            elements = node
            keys = list(map(lambda el: f"[{el}]", keys))
        except Exception:
            pass

    if keys is None:
        raise NotImplementedError(f"Type {type(node)} is not supported.")

    def index(x, leaves, mask):
        return next(
            i
            for i, (leaf, m) in enumerate(zip(leaves, mask))
            if x is leaf and m is True
        )

    names = []
    # Mask is used to deal with duplicates values in the elements
    mask = [True for _ in elements]
    for leaf in leaves:
        i = index(leaf, elements, mask)
        names.append(keys[i])

        mask[i] = False

    return zip(leaves, names)


# Computes whether the given element is a pytree (i.e., has children)
# or a leaf.
def is_tree(_any: Any):
    leaves = jtu.tree_leaves(_any)

    return (len(leaves) != 1) or (leaves[0] is not _any)


# Function used to create a mask of a pytree.
# Compared to jax.tree_map, it keep tracks of a state that is passed
# down, from root to leaves.
def tree_mask(node, mask_fn: Callable, root_state=None, is_leaf=None, name=None):
    # Determine whether the node is a leaf
    leaf = not is_tree(node) or (is_leaf is not None and is_leaf(root_state, node))

    # Apply the mask function to the node
    if name is not None:
        root_state = mask_fn(node, root_state, leaf, name)
    else:
        root_state = mask_fn(node, root_state, leaf)

    if leaf:
        return root_state

    leaves, structure = jtu.tree_flatten(node, is_leaf=lambda leaf: leaf is not node)

    if name is not None:
        mask = [
            tree_mask(
                child,
                mask_fn,
                copy.deepcopy(root_state),
                is_leaf,
                filter=filter,
                name=key,
            )
            for (child, key) in _get_named_leaves(node, leaves)
        ]
    else:
        mask = [
            tree_mask(child, mask_fn, copy.deepcopy(root_state), is_leaf)
            for child in leaves
        ]

    return jtu.tree_unflatten(structure, mask)
