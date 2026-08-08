# Open questions on the defect triage

This file is temporary. It holds the defects that are **not yet settled**, so that
the discussion has one place to live. When the last one is resolved, the file is
deleted and tracking continues as GitHub issues plus the `bug`-marked tests: a
defect is fixed when its tests go green and its issue closes.

Everything already agreed is gone from here. Those 19 defects are now
[#67](https://github.com/liukidar/pcx/issues/67) to
[#79](https://github.com/liukidar/pcx/issues/79), and each `bug` marker names the
issue that owns it, so `grep '#70'` finds every test guarding that fix.

Ten items remain, in four groups. Each still has a failing test asserting the correct
behaviour, except 2, whose test has been deleted because we now agree it was asserting
a non-requirement. Run them with `just test-bugs`; each marker names its `BUGS.md#N`.
The numbers are the original triage numbers, kept so they still match the annotated
list.

---

## Where I think Luca is right

### 2. Shared submodules are double-counted in the energy

**Withdrawn as a defect.** `BaseModule.submodules` yields one module once per
reference rather than once per unique object, so a module reachable through two
attributes contributes its energy twice. Luca's position is that the library was
designed on the assumption that a module is reachable exactly once, which makes
this undefined rather than wrong. Agreed, and the test asserting deduplication has
been deleted.

Two things still to settle:

1. **Where does the precondition get written down?** It is currently nowhere, and
   nothing raises when it is violated. README, the `Module` docstring, or a check
   that raises when the same module is reached twice.
2. **`tree_ref` deduplicates on `id()`.** That is what makes `pxnn.shared` work, so
   the two traversals genuinely disagree about what a repeated reference means.
   Shared *parameters* are supported and are being fixed in
   [#73](https://github.com/liukidar/pcx/issues/73); shared *modules* are not. Is
   that split intentional, and should it be stated?

*Test:* deleted, with a comment in `tests/core/test_module.py` recording why.

### 11. `zero_energy` ignores the node's shape

**Severity was overstated on my side.** I wrote that unconstrained nodes are
"unusable at any batch size ≠ 1". That is wrong, and Luca's "I have used batching
all the time" is exactly right. Measured:

```
se_energy   via __call__   -> shape ()
se_energy   via set()      -> shape (3,)
zero_energy via __call__   -> shape ()
zero_energy via set()      -> TypeError: cannot reshape (1,) into (3, -1)
zero_energy inside vmap    -> shape (3,)
```

`zero_energy` works under `vmap`, which is how every tutorial drives a model. It
only breaks on the `set()` path, and it breaks there for the same reason as 19
below: `Vode.energy` picks its branch from `self.shape`, which only `__call__`
records. **This is a symptom of 19, not an independent defect**, and it is Low, not
Medium.

The narrow claim still stands: `zero_energy` is documented as interchangeable with
`se_energy` and `ce_energy`, both of which return one term per element of `h`, and a
fixed `(1,)` is not interchangeable. Fixing it is a one-liner and harmless, but it
should probably wait for the decision on 19.

*Test:* `tests/numerics/test_energy.py::test_zero_energy_has_the_same_shape_as_the_node_value`
*Source:* `pcx/predictive_coding/_energy.py`

### 7. A cleared `ParamDict` raises on every read

**Partly conceded, and the remaining disagreement is narrow.**

Luca: "I think `None` had a special meaning, like to signal you cannot access the
value, so the exception there may be intentional. I have never seen that causing
issues." Both halves are fair. You rarely hit it because `__setitem__` re-creates
`{}` on the first write, so a forward pass after a clear repopulates the cache
before anything reads it.

But the code does not implement that intent, it leaks an internal:

- `"E" in cache` → `TypeError: argument of type 'NoneType' is not iterable`
- `cache["E"]` → `TypeError: 'NoneType' object is not subscriptable`
- `cache.get("E", default)` → `AttributeError: 'NoneType' object has no attribute 'get'`

**Question:** if `None` means "unreadable", should reads raise a real pcx error with
a message saying so, instead of a `TypeError` from `NoneType`?

And regardless of that: `__contains__` returning `False` and `.get(key, default)`
returning the default seem indefensible to break, since `.get` with a default exists
precisely to be safe on absent keys. `__getitem__` is the one where your reading is
strongest, and that decides whether one of the four tests gets deleted.

*Tests:* three in `tests/core/test_parameter.py` (`*_cleared_paramdict_*`),
`tests/numerics/test_vode.py::test_a_cleared_cache_reads_as_empty_rather_than_raising`
*Source:* `pcx/core/_parameter.py`, `ParamDict`

---

## Where I think we should wait

### 19. `Vode.energy()` returns a different shape depending on how `u` was set

Luca: "indeed they should be equivalent, but I would wait to fix to understand if
there is a reason for the difference." There is one, and it is load-bearing.

`Vode.__call__` is documented as equivalent to `vode.set("u", u).get("h")`, but it
also does `self.shape.set(h.shape)` (`_vode.py:224`). `Vode.energy` then branches on
`self.h.shape == self.shape` (`_vode.py:321`) to choose between a per-sample vector
and a scalar sum. That comparison is **the library's only mechanism for detecting a
`vmap` context.** So `set()` not recording the shape is not an oversight to patch;
aligning the two paths would break vmap detection.

Measured, outside `vmap`: `__call__` gives `()`, `set()` gives `(3,)`. Totals agree,
per-sample resolution is lost.

Fixing this properly means replacing the heuristic rather than patching either path,
which is exactly the "what does jax 0.11 give us for free now" question raised
against the old entry 4. **No issue opened; this needs a design decision first.**

*Test:* `tests/numerics/test_vode.py::test_energy_is_the_same_whether_u_was_set_by_call_or_by_set`
*Source:* `pcx/predictive_coding/_vode.py`

---

## Where I explained it badly

The claims here stand; the write-ups did not carry them. Repros instead of prose.

### 6. `value_and_grad` writes back to positionally-passed params

Luca: "not sure, would have to look more at it." Fair, it is subtle.

The library's protocol is that positional arguments are pure jax values and are not
tracked, while keyword arguments are. `ValueAndGrad._t` forwards `*args` straight
into `jax.value_and_grad`, which leaves non-differentiated positional arguments as
the exact objects passed in. So mutating one inside the function mutates the
caller's object.

The dangerous case is when the mutation involves the differentiated value: the
caller's positional `Param` is left holding **a live autodiff tracer**. Nothing
raises at the time. It detonates later as an `UnexpectedTracerError` in unrelated
code, which is close to undebuggable.

`Jit` gets this right, so the same user code behaves differently depending on which
transform wraps it. That inconsistency is the part worth deciding on.

*Tests:* `tests/functional/test_transforms.py::test_value_and_grad_does_not_write_back_a_param_passed_positionally`,
`::test_value_and_grad_does_not_leak_a_tracer_into_a_positional_param`
*Source:* `pcx/functional/_transform.py`, `ValueAndGrad._t`

### 9. `pxu.step` does not restore status and has no `try/finally`

Luca: "mmh, not sure what it means." My entry tangled two separate claims. Split:

**(a) No `try/finally`.** An exception in the body skips both the cache clear and
the status reset, so the model is left corrupted. Under pytest, the next test
inherits it. This one seems indefensible regardless of the design intent.

**(b) A scalar status is never restored.** The reset only runs when a 2-tuple is
passed (`_misc.py:66`), so `with pxu.step(model, STATUS.INIT):` leaves the model in
`'init'` afterwards, and any energy computed before the next block runs under the
wrong ruleset.

(b) may well be the intended design: you set the status each time, so restoring is
unnecessary. The case that suggests otherwise is nesting, where an inner block's
exit leaves the outer block in the wrong status:

```python
with pxu.step(model, STATUS.NONE):
    with pxu.step(model, STATUS.INIT):
        ...
    # model is in 'init' here, not 'none'
```

**Question:** is (b) intended, and is nesting meant to work? If (b) is intended, one
of the three tests should go and only (a) and the nesting case remain.

*Tests:* three in `tests/utils/test_step.py`
*Source:* `pcx/utils/_misc.py`, `step`

### 20. The `Scan` docstring example cannot produce its documented output

Luca: "not sure, need more context." There is not much more; it is just a wrong
annotation on the library's only executable illustration of the transform.

```python
def f(x, count):
    count = count + x
    return (count + x,), None


Scan(f, xs=jax.numpy.arange(5))(0)  # [0, 1, 3, 6, 10], None   <- documented
# ((20,), None)            <- actual
```

Two reasons it cannot match. The body adds `x` twice per step, so the running total
is 0, 2, 6, 12, 20 rather than 0, 1, 3, 6, 10. And the first element of the return is
the final carry, not a per-step sequence, so no scan of any body could produce a
five-element list there. A reader copying this to learn `Scan` learns the wrong
signature.

**Question:** should the example be corrected to match `Scan`, or should the
docstring show the `ys` return that does produce a sequence?

*Test:* `tests/functional/test_flow.py::test_scan_docstring_example_produces_its_documented_output`
*Source:* `pcx/functional/_flow.py`, `Scan`

---

## No response yet

These three were not covered in the reply. All small, all with failing tests.

### 12. `pcx.set` cannot be called on a plain value

`set` on a non-param executes `obj = set(x)`, calling itself with one argument:
`TypeError: set() missing 1 required positional argument: 'x'`. The helper exists so
call sites need not know whether they hold a param, and that is the one path that
cannot work.

*Test:* `tests/core/test_parameter.py::test_set_on_a_plain_value_returns_the_new_value`

### 13. `tree_inject` cannot inject plain values

It calls `.get()` on every element of `values`, contradicting both the documented
type ("input sequence of values to inject") and the default
`inject_fn = lambda n, v: n.set(v)`. Arrays coming straight out of a transform cannot
be injected: `AttributeError: 'ArrayImpl' object has no attribute 'get'`.

*Test:* `tests/core/test_tree.py::test_inject_accepts_plain_values`

### 14. `tree_inject` raises bare `StopIteration` when given too few values

Surplus values raise a clear `ValueError`; a shortfall raises `StopIteration` from
`next()` inside the traversal. Under PEP 479 that becomes an unrelated `RuntimeError`
inside a generator, and the params visited before exhaustion have already been
overwritten.

*Test:* `tests/core/test_tree.py::test_inject_strict_rejects_missing_values`

---

## Already closed out

**Entry 4, `pxf.vmap` on jax >= 0.4.34, was fixed upstream in v0.6.3.** `Vmap._t`
called `_process_mask` without an `rkg_mask`, leaving `mask["__RKG"]` as `None`,
which jax has rejected as a prefix since 0.4.34. Bisected as working on 0.4.33 and
failing on every release after, which meant no tutorial notebook ran on a modern
jax. Fixed by passing `is_leaf=lambda mask: mask is None` to the mask `tree_map`.
The nine tests that covered it now pass and have had their markers removed.

The three findings investigated and cleared as **not** defects have moved to
[CONTRIBUTING.md](CONTRIBUTING.md), along with the mutation-testing evidence, since
both outlive this file.
