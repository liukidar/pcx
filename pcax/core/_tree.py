__all__ = ["tree_extract", "tree_inject", "tree_ref", "tree_unref"]


from typing import Any, Tuple, Sequence, Callable
from jaxtyping import PyTree

import jax.tree_util as jtu
import equinox as eqx

from ..core._parameter import BaseParam, DynamicParam
from ..core._static import StaticParam


########################################################################################################################
#
# TREE
#
# A set of helper functions to simplify the management of stateful pytrees (and pydags) based on 'AbstracParams'.
# See the documentation of each function for more details.
#
########################################################################################################################

# Utils ################################################################################################################


def _cache() -> Callable[[Any], int | None]:
    """Utiliy function to create a fast cache set, which keeps track of the order in which elements are seen.
    Usage:

    ```
    cache = _cache()

    if (n := cache(x)) is not None:
        print(f"{x} was already observed. It is the {n}th unique observed object")
    ```

    Returns:
        Callable[[Any], int | None]: callable cache set.
    """
    _data = {}
    _setdefault = _data.setdefault
    _n = 0

    def _add(x: Any) -> int | None:
        """Cache 'test and set' function.

        Args:
            x (Any): element to check against the cache

        Returns:
            int | None: None if x is a newly seen object; otherwise i,
                where i is the number of unique objects seen before x.
        """
        nonlocal _n
        _r = _setdefault(x, _n)
        if _r == _n:
            _n += 1

            return None
        else:
            return _r

    return _add


class _BaseParamRef(StaticParam):
    """
    Class used to replace multiple references to the same BaseParam with a static index
    to the unique BaseParam. This allows to transform pydags to pytrees (as duplicate direct
    references are replaced with indices).
    """

    def __init__(self, n: int) -> None:
        """_BaseParamRef constructor.

        Args:
            n (int): the index of the referenced parameter as leaf in the flattened pytree it belongs to.
        """
        super().__init__(n)


# Core #################################################################################################################


def tree_apply(
    fn: Callable[[Any], None], filter_fn: Callable[[Any], bool], tree: PyTree, recursive: bool = True
) -> None:
    """Executes a function on the selected nodes of the pytree. Note that pydag are supported since the structure of
    the pytree is preserved (i.e., the function can only modify the content of the nodes, not the nodes themselves).
    This, however, implies that if a duplicate reference is present in the pytree, the function will be applied to each
    occurrence of the reference (so multiple times on the same node), which must be taken into account when designing
    the function. For example:

    ```python
    p = Param(1.0)

    m = [p, p]

    def inc(p):
        p += 1

    tree_apply(inc, lambda x: isinstance(x, Param), m)

    print(m)  # [Param(3.0), Param(3.0)]
    ```

    Args:
        fn (Callable[[Any], None]): function to apply to the selected nodes of the pytree.
        filter_fn (Callable[[Any], bool]): filter function to select the nodes on which to apply 'fn'.
        tree (PyTree): input pytree.
        recursive (bool, optional): whether to call 'fn' recursively or to stop after the first generation of nodes
            matching 'filter_fn' is encountered. Normally is set to False for performance reasons when targeting
            parameters (that are leaves of the pytree).
    """

    def _wrap_fn(x):
        if r := filter_fn(x):
            fn(x)
        return r

    leaves = jtu.tree_leaves(tree, is_leaf=_wrap_fn)

    if recursive:
        for leaf in leaves:
            for x in eqx.tree_flatten_one_level(leaf)[0]:
                if x is not tree:
                    tree_apply(fn, filter_fn, tree=x)


def tree_extract(
    pydag: PyTree,
    *rest: ...,
    extract_fn: Callable[[Any | Tuple[Any, ...]], Any] = lambda x: x,
    filter_fn: Callable[[Any], bool] = lambda x: isinstance(x, DynamicParam),
    is_pytree: bool = False,
) -> Sequence[Any]:
    """Extract an ordered sequence of values from the BaseParams of a pytree.
    Similarly to 'ref'/'unref', 'extract'/'inject' rely on a consistent structure of the input pytree
    (i.e., you can only inject into the same pytree structure you extracted from).

    Args:
        pydag (PyTree): input pydag.
        rest: (...): a tuple of pytrees, each of which has the same structure as tree or has tree as a prefix.
        extract_fn (Callable[[Any | Tuple[Any, ...]], Any], optional): function that takes 1 + len(rest) arguments,
            to be applied at the corresponding leaves of the pytrees.
        filter_fn (Callable[[Any], bool], optional): filter function to select the BaseParam
            on which to apply 'extract_fn'.
        is_pytree (bool, optional): whether the input pydag is a pytree and contains no references; used to avoid
            unnecessary reffing.

    Returns:
        Sequence[Any]: list of extracted values.
    """
    assert is_pytree is True, "Not implemented for non-pytrees."

    # We use jtu.tree_map to apply extract_fn to the dynamically identified leaves of the pytree.
    _values = []

    def _map_fn(x, *rest):
        if filter_fn(x):
            _values.append(extract_fn(x, *rest))

        return x

    jtu.tree_map(_map_fn, pydag, *rest, is_leaf=lambda x: isinstance(x, BaseParam))

    return tuple(_values)


def tree_inject(
    pydag: PyTree,
    *,
    params: PyTree = None,
    values: Sequence[Any] = None,
    inject_fn: Callable[[Tuple[Any, Any]], None] = lambda n, v: n.set(v),
    filter_fn: Callable[[Any], bool] = lambda x: isinstance(x, DynamicParam),
    is_pytree: bool = False,
    strict: bool = True,
) -> PyTree:
    """Inverse function of 'extract'. Note that it doesn't modify the pydag structure, but rather the values
    of its BaseParam leaves.

    Args:
        pydag (PyTree): input pydag.
        values (Sequence[Any]): input sequence of values to inject into pydag at the selected leaves.
        inject_fn (Callable[[Tuple[Any, Any]], None], optional): function that takes the target leaf and
            previously extracted value to inject into the leaf. Note: the return value is ignored and does
            not replace the original leaf as in 'jtu.tree_map'.
        filter_fn (Callable[[Any], bool], optional): filter function to select the leaves on which to apply 'extract_fn'
        is_pytree (bool, optional): whether the input pydag is a pytree and contains no references; used to
            avoid unnecessary reffing.
        strict (bool, optional): if True, the number of values must match the number of leaves in the pytree.

    Returns:
        PyTree: pytree with values injected via 'inject_fn'.
    """
    assert is_pytree is True, "Not implemented for non-pytrees."

    if values is None:
        values = filter(filter_fn, jtu.tree_leaves(params, is_leaf=lambda x: isinstance(x, BaseParam)))
    else:
        assert params is None, "Cannot specify both 'values' and 'params'"

    # We use jtu.tree_leaves to apply inject_fn to the dynamically identified leaves of the pytree.
    _values_it = iter(values)

    def _inject_param(x: Any):
        if isinstance(x, BaseParam):
            if filter_fn(x):
                inject_fn(x, next(_values_it).get())

            return True
        else:
            return False

    jtu.tree_leaves(pydag, is_leaf=_inject_param)

    if strict is True:
        # This is to assert the user didn't mess up with the pytree structure.
        try:
            next(_values_it)
            raise ValueError("The number of values does not match the number of leaves in the pytree.")
        except StopIteration:
            pass

    return pydag


def tree_ref(pydag: PyTree) -> PyTree:
    """Transforms a pydag in a pytree by replacing all duplicate BaseParams references with explicit indexing.
    This effectively means that all the occurences, except the first encountered, of each unique parameter are replaced
    by an integer index wrapped into a _BaseParamRef.
    This is necessary as jax treats all input/output values of its transformations as pytree, which results in
    unexpected behaviour when passing in pydags.

    NOTE #1: ref has some usage limitations, see unref for a complete overview.

    Args:
        pydag (PyTree): input pydag

    Returns:
        PyTree: output pytree with duplicate BaseParams replaced by explicit references.
    """
    # We use BaseParam and not Param to target also StaticParams and ParamDicts.
    _seen = _cache()

    def _ref(x):
        if isinstance(x, _BaseParamRef):
            # We ref a ref to keep the structure consistent and allow multiple reffing/unreffing.
            return _BaseParamRef(x)
        elif isinstance(x, BaseParam) and ((_ref := _seen(id(x))) is not None):
            # If the parameter is already seen, we replace it with a reference.
            # _ref is an integer that refers to the index of parameter in the ordered sequence of unique parameters
            # encountered in pydag during flattening.
            return _BaseParamRef(_ref)
        return x

    return jtu.tree_map(_ref, pydag, is_leaf=lambda x: isinstance(x, BaseParam))


def tree_unref(pytree: PyTree) -> PyTree:
    """Replace explicit _BaseParamRef with the indexed BaseParam, recreating the original pydag.
    The most common usage pattern would be the following:

    ```python
    def f(pytree):
        pydag = unref(pytree)
        return ref(pydag)

    p = pydag(...)
    t = a_jax_transformation(f)
    p = unref(t(ref(p)))
    ```

    This is automatically and efficiently done when using automatic parameter tracing via pcax transformations
    (i.e., passing parameters within the kwargs of a pcax transformation).

    NOTE #1: Refernces work via simple indexing, which requires the underlying pydag/pytree structure to be constant
    between ref and unref (i.e., unref has defined behaviour only if used on a pytree with the same structure as
    the value returned by ref). For example, the following is not allowed:

    ```python
    p = pydag()
    pytree = ref(p)
    a, p = unref([Param(), pytree])  # THIS IS WRONG: pytree has not shape [Param(), pytree]
    ```

    NOTE #2: Note #1 implies that it is possible to ref an already [partially] reffed pytree, but unreffing must be
    done in the same (reversed) order:

    ```python
    # Example 1
    p = unref(unref(ref(ref(p))))

    # Example 2
    p1 = pydag()
    tree1 = ref(p1)
    p2 = [pydag(), tree1]
    tree2 = ref(p2)

    # Here the following is NOT allowed (as the structure of p2[1] may be changed by the second reffing):
    p1 = unref(p2[1])  # WRONG!

    # Instead the following order must be respected:
    p2 = unref(tree2)
    tree1 = p2[1]
    p1 = unref(tree1)
    ```

    NOTE #3: the behaviour of NOTE #2 has not been extensively tested, so not sure which are the exact limitations of
    the approach.

    Args:
        pytree (PyTree): input pytree

    Returns:
        PyTree: output pydag with resolved references.
    """
    _seen = []

    def _unref(x: Any) -> Any:
        if isinstance(x, _BaseParamRef):
            # Since a _BaseParamRef can be nested, we check if we reached the final index and, if so, unref.
            x = x.get()
            return _seen[x] if isinstance(x, int) else x
        elif isinstance(x, BaseParam):
            _seen.append(x)

        return x

    return jtu.tree_map(_unref, pytree, is_leaf=lambda x: isinstance(x, BaseParam))
