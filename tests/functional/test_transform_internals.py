"""Internals of `pcx.functional._transform`: representation, mask processing, vmap.

`tests/functional/test_transforms.py` covers what the transforms *compute*. This
file covers the machinery around that: how a transform describes itself, how a
`kwargs_mask` is expanded into a per-leaf mask, and the vmap-specific bookkeeping
(batch-size inference, `out_axes`, and splitting the global random key across
lanes).

The oracles are, in order of preference: the documented behaviour of the method
(`_process_mask`'s docstring spells out both the tuple-key expansion and the
callable-leaf rule); the equivalent raw-jax computation; and, for the random key,
the documented contract of `RKGState.split` — `jax.random.split(key, n + 1)`,
keeping `[0]` as the new state and handing out `[1:]`. No expected value is taken
from what pcx returns.

Note on vmap: these tests were written while `pxf.vmap` was broken on jax >= 0.4.34,
as the specification a fix would have to satisfy. v0.6.3 fixed it, so they now pass
and guard the fix.
"""

import functools

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from conftest import assert_allclose

import pcx
import pcx.functional as pxf
import pcx.utils as pxu


def diff_params():
    """Mask selecting every `pcx.Param` for differentiation.

    A fresh one per call, because `.to()` mutates the mask in place.
    """
    return pxu.M(pcx.Param).to((False, True))


def scaled(x, factor=2.0):
    """A module-level function with a default argument, used as a repr subject."""
    return x * factor


def _affine_loss(w, x):
    """Reference loss over plain arrays, differentiable by raw jax."""
    return jnp.sum(jnp.tanh(x * w + 1.0))


class _Scaler:
    """A callable object — a perfectly ordinary thing to hand to a transform."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor


# Representation #######################################################################


def test_repr_of_a_transform_names_the_wrapped_function():
    """`_repr_function` is documented as a "human readable function representation",
    and `__repr__` embeds it as `Jit(fn=...)`.

    A transform is an opaque object at a debugger prompt or in a traceback frame; the
    only thing that tells the user *which* of their functions it wraps is this
    string. `pxf.jit()` is applied to a dozen functions in a typical training script.
    """
    assert repr(pxf.jit()(_affine_loss)) == "Jit(fn=_affine_loss)"


def test_repr_reports_the_default_arguments_of_the_wrapped_function():
    """Default arguments are baked into a jitted function at trace time, so which
    defaults were in force is exactly the kind of thing the repr exists to show."""
    text = repr(pxf.jit()(scaled))

    assert text.startswith("Jit(fn=scaled"), text
    assert "factor=2.0" in text, text


def test_repr_of_a_partial_names_the_underlying_function_and_its_bound_value():
    """`functools.partial` has no `__name__`, so `_repr_function` walks `.func` to
    find one. Partials are the standard way to pin a hyperparameter before handing a
    function to a transform, and a repr of `functools.partial` would name none of
    the three things the user cares about."""
    text = repr(pxf.jit()(functools.partial(scaled, factor=3.0)))

    assert "scaled" in text, text
    assert "factor=3.0" in text, text


def test_repr_of_a_callable_object_names_its_class():
    """A callable object has neither `__name__` nor `.func`, so the class name is the
    only human-readable handle available — and `_repr_function` falls back to it."""
    assert repr(pxf.jit()(_Scaler(2.0))) == "Jit(fn=_Scaler)"


@pytest.mark.bug(
    "#77: _BaseTransform.__init__ collapses __wrapped__ to the innermost function, so repr() of a composed transform "
    "hides every intermediate layer and the recursive branch in __repr__ is unreachable"
)
def test_repr_of_a_nested_transform_names_every_layer():
    """A composed transform must describe the whole stack, not just the innermost
    function.

    `__repr__` contains a branch for exactly this — `repr(self.__wrapped__) if
    isinstance(self.__wrapped__, _BaseTransform)` — so nested reporting is plainly
    the intent. But `__init__` sets `self.__wrapped__ = _fn.__wrapped__` when it
    wraps another transform, which by induction is always the innermost plain
    function, so that branch can never run.

    The consequence is that `pxf.jit()(f)` and `pxf.jit()(pxf.value_and_grad(m)(f))`
    print identically while computing entirely different things: one returns a value,
    the other a `(value, gradients)` pair. Composition is the documented way to build
    a training step, so this is the object a user most needs to identify.
    """
    composed = pxf.jit()(pxf.value_and_grad(diff_params())(_affine_loss))

    assert "ValueAndGrad" in repr(composed), repr(composed)


# Mask processing ######################################################################


def test_a_tuple_of_names_as_a_mask_key_applies_the_mask_to_each_of_them():
    """Documented in `_process_mask`: "If the mask keys are tuples, they are expanded
    into individual keys".

    `{("model", "optim"): mask}` is how a user avoids repeating one mask across
    several kwargs. If the expansion dropped a key, `jtu.tree_map` would reject the
    mask outright; if it applied the mask to the wrong key, the wrong parameters
    would be differentiated — and a gradient computed against the wrong subset is
    still a perfectly plausible-looking number.
    """
    w = jnp.array([0.5, -1.25])
    v = jnp.array([2.0, 0.5])
    x = jnp.array([2.0, 3.0])
    p, q = pcx.Param(w), pcx.Param(v)

    @pxf.value_and_grad({("p", "q"): pxu.M(pcx.Param).to((False, True))})
    def loss(x, *, p, q):
        return _affine_loss(p.get(), x) + jnp.sum(q.get() ** 2)

    def reference(w, v, x):
        return _affine_loss(w, x) + jnp.sum(v**2)

    value, grads = loss(x, p=p, q=q)
    expected_value, (expected_dw, expected_dv) = jax.value_and_grad(reference, argnums=(0, 1))(w, v, x)

    assert_allclose(value, expected_value)
    assert_allclose(grads["p"].get(), expected_dw)
    assert_allclose(grads["q"].get(), expected_dv)


def test_a_single_argument_callable_mask_leaf_is_applied_to_its_kwarg_subtree():
    """Documented in `_process_mask`: "If a callable object is given, then it is
    applied to the corresponding kwarg subtree and the result is used as the mask".

    `map_fn` first tries the two-argument form `mask(kwarg, is_pytree=True)`, which
    exists so that `pxu.M` can skip re-reffing an already-reffed tree, and falls back
    to the plain one-argument call. A user-written mask is an ordinary one-argument
    function, so the fallback is the path most user code takes.
    """
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)
    frozen = pcx.Param(jnp.array([9.0, 9.0]))

    def differentiate_everything(subtree):
        """One positional argument only — the documented mask-leaf signature."""
        return jtu.tree_map(lambda _: True, subtree, is_leaf=lambda leaf: isinstance(leaf, pcx.BaseParam))

    def differentiate_nothing(subtree):
        return jtu.tree_map(lambda _: False, subtree, is_leaf=lambda leaf: isinstance(leaf, pcx.BaseParam))

    @pxf.value_and_grad({"p": differentiate_everything, "q": differentiate_nothing})
    def loss(x, *, p, q):
        return _affine_loss(p.get(), x) + jnp.sum(q.get())

    value, grads = loss(x, p=param, q=frozen)

    assert_allclose(value, _affine_loss(w, x) + 18.0)
    assert_allclose(grads["p"].get(), jax.grad(_affine_loss)(w, x))
    assert grads["q"] is None, "a mask leaf returning False must exclude the parameter from the gradient"


@pytest.mark.bug(
    "#78: _process_mask's `except TypeError` fallback swallows a TypeError raised inside a user's mask callable and "
    "calls it a second time, so a broken mask runs twice and reports a confusing chained error"
)
def test_a_mask_callable_that_raises_a_type_error_is_not_invoked_twice():
    """A user's mask callable must be invoked exactly once per subtree.

    `map_fn` uses `try: mask(kwarg, is_pytree=True) / except TypeError: mask(kwarg)`
    to discover the callable's arity. `except TypeError` cannot tell "this callable
    does not accept `is_pytree`" from "this callable ran and raised `TypeError`", so
    a mask that accepts the keyword and then fails — a typo, a bad `isinstance`, an
    array operation on a `None` leaf — is silently re-executed.

    Two consequences, both real: any side effect in the mask (a counter, a log line,
    a cache write) happens twice, and the traceback the user finally sees is the
    *second* failure with the first hanging off it as "During handling of the above
    exception, another exception occurred", pointing at `_process_mask` rather than
    at their own code.
    """
    calls = []

    def broken_mask(subtree, is_pytree=False):
        calls.append(is_pytree)
        raise TypeError("mask exploded")

    param = pcx.Param(jnp.array([0.5, -1.25]))

    @pxf.value_and_grad({"p": broken_mask})
    def loss(x, *, p):
        return _affine_loss(p.get(), x)

    with pytest.raises(TypeError, match="mask exploded"):
        loss(jnp.array([2.0, 3.0]), p=param)

    assert len(calls) == 1, f"the failing mask callable was invoked {len(calls)} times: {calls}"


def test_value_and_grad_with_a_tuple_argnums_returns_positional_and_keyword_gradients():
    """`argnums` asks for gradients of positional arguments as well, and pcx appends
    its own kwargs argnum to whatever the user gave.

    The result is repacked as `(positional_gradients, kwarg_gradients)`. Both halves
    have to be right: a mis-sliced repack would hand the optimiser the gradient with
    respect to the *input data* as if it were the gradient of the weights.

    (The documented `argnums=0` integer spelling is rejected outright — #74 —
    so this uses the `(0,)` tuple form that does work.)
    """
    w = jnp.array([0.5, -1.25])
    x = jnp.array([2.0, 3.0])
    param = pcx.Param(w)

    @pxf.value_and_grad(diff_params(), argnums=(0,))
    def loss(x, *, p):
        return _affine_loss(p.get(), x)

    value, (positional_grads, kwarg_grads) = loss(x, p=param)
    expected_value, expected_dx = jax.value_and_grad(lambda x, w: _affine_loss(w, x))(x, w)

    assert_allclose(value, expected_value)
    assert len(positional_grads) == 1, "argnums=(0,) asks for exactly one positional gradient"
    assert_allclose(positional_grads[0], expected_dx)
    assert_allclose(kwarg_grads["p"].get(), jax.grad(_affine_loss)(w, x))


# vmap #################################################################################
#
# `Vmap._t` used to call `_process_mask` without an `rkg_mask`, leaving
# `mask["__RKG"]` as None while the kwargs held a real `RandomKeyGenerator`, which
# jax >= 0.4.34 rejects. v0.6.3 fixed it with `is_leaf=lambda mask: mask is None`.
# These tests pin the batching, out_axes and per-lane key behaviour it must keep.


def lane_keys(seed: int, n: int):
    """The `n` per-lane keys `Vmap._t` must hand to the lanes.

    Straight from `RKGState.split`: `jax.random.split(key, n + 1)`, element 0 becomes
    the new state and elements `1:` are handed out.
    """
    return jax.random.split(jax.random.PRNGKey(seed), n + 1)[1:]


def draw_from(key):
    """One standard normal drawn the way `pcx.RKG()` draws it inside a lane.

    `RKG()` is `split(1)`, i.e. `jax.random.split(key, 2)` keeping `[0]` as the new
    state and returning `[1]` to the caller.
    """
    return jax.random.normal(jax.random.split(key, 2)[1], ())


def test_vmap_honours_a_non_zero_out_axes():
    """`out_axes` is forwarded to `jax.vmap`, so it must place the mapped axis where
    the user asked.

    pcx wraps the user's value in `(out_axes, kwargs_mask)` to carry the tracked
    kwargs alongside the result; if the user's entry were dropped or defaulted, the
    output would come back transposed — a silent axis swap that later code will
    happily broadcast against.
    """

    @pxf.vmap(in_axes=(0,), out_axes=1)
    def rowwise(row):
        return row * 2.0

    xs = jnp.arange(6.0).reshape(3, 2)

    assert_allclose(rowwise(xs), jax.vmap(lambda r: r * 2.0, in_axes=0, out_axes=1)(xs))


def test_vmap_infers_the_batch_size_from_the_axis_named_by_in_axes():
    """`_extract_vaxes_dim` reads `param.shape[in_axes_entry]` to learn how many
    lanes there are, because the global key has to be split into exactly that many.

    Mapping over axis 1 of a `(2, 5)` array is five lanes, not two. Inferring the
    wrong number either crashes inside `jax.vmap` or — worse — splits a key of the
    wrong length and gives every lane a correlated stream.
    """
    pcx.RKG.seed(0)

    @pxf.vmap(in_axes=(1,), out_axes=0)
    def draw(column):
        return column.sum() + jax.random.normal(pcx.RKG(), ())

    out = draw(jnp.zeros((2, 5)))

    assert out.shape == (5,), f"mapping over axis 1 of a (2, 5) array gives 5 lanes, got shape {out.shape}"
    assert_allclose(out, jnp.stack([draw_from(k) for k in lane_keys(0, 5)]))


def test_vmap_gives_each_lane_an_independent_slice_of_the_global_key():
    """The global `RKG` key is split across the mapped axis so every lane draws its
    own randomness.

    This is the whole reason `Vmap._t` computes a batch size at all. Without the
    split, `jax.vmap` would broadcast one key to every lane and a vmapped dropout
    mask — or a vmapped weight initialisation — would be *identical* for every
    example in the batch. Nothing raises; the model simply trains against a batch of
    duplicated noise.
    """
    pcx.RKG.seed(0)

    @pxf.vmap(in_axes=(0,), out_axes=0)
    def draw(x):
        return x + jax.random.normal(pcx.RKG(), ())

    samples = draw(jnp.zeros(4))

    assert len(set(samples.tolist())) == 4, f"lanes drew duplicate samples: {samples}"
    assert_allclose(samples, jnp.stack([draw_from(k) for k in lane_keys(0, 4)]))


def test_vmap_restores_a_single_unbatched_key_in_the_global_rkg():
    """After the call the global key must be one ordinary key again.

    `Vmap._t` widens `RKG.key` to `(n, 2)` for the duration of the call and narrows
    it back with `key[0]` on the way out. If the batched key escaped, every later
    `jax.random.*` call in the process would receive a stack of keys instead of one
    and either raise or silently draw a batch of samples where a single value was
    expected.
    """
    pcx.RKG.seed(0)

    @pxf.vmap(in_axes=(0,), out_axes=0)
    def draw(x):
        return x + jax.random.normal(pcx.RKG(), ())

    draw(jnp.zeros(4))

    key = pcx.RKG.key.get()

    assert key.shape == (2,), f"the global key came back with shape {key.shape}, not a single key"
    # Lane 0's state after its own draw: split(lane_key, 2)[0].
    assert_allclose(key, jax.random.split(lane_keys(0, 4)[0], 2)[0])
