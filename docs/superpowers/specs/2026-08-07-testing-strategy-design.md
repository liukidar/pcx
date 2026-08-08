# PCX Testing Strategy

**Date:** 2026-08-07
**Status:** design, awaiting review

## Context

PCX has no tests. It has ~1040 statements across five subpackages and, as of this
document, 18% line coverage from a handful of smoke tests added alongside the
tooling migration.

Two defects found while surveying the codebase set the terms for everything below.

**`pxf.vmap` is broken on every jax release from 0.4.34 onward.** Verified by
bisection across 0.4.33, 0.4.34, 0.4.35, 0.5.3, 0.6.2, 0.7.2 and 0.11.0. The
`Vmap` transform leaves `mask["__RKG"]` as `None`; jax treats `None` as an empty
pytree node and since 0.4.34 rejects it as a prefix against a non-`None` subtree.
Every tutorial notebook uses `pxf.vmap`, so none of them run on a modern jax. The
declared constraint was `jax = "^0.4.33"`, which permits newer versions, but the
committed `poetry.lock` pinned exactly 0.4.33 — so the authors' environments
worked while every fresh install was broken.

**Two subpackages failed to import at all**, after `UP035` rewrote
`from typing import Callable` to `from collections.abc import Callable`.
`typing.Callable` tolerates a forward-reference string in a `|` union;
`collections.abc.Callable` does not, so `"_BaseTransform" | Callable` raised at
import. Fixed by quoting the full annotations, with `tests/test_imports.py` added
to prevent recurrence.

Both defects share a root cause that this strategy has to answer for: **there was
no test that ran the library the way a user runs it.** A lockfile-only CI is
structurally incapable of catching the first, and a smoke test that imports only
the top-level package is incapable of catching the second.

## Governing principle

Every expectation is derived from what the code is *for* — the mathematics of
predictive coding, the documented contract, the behaviour of the jax primitive
being wrapped — and never from what the implementation currently returns.

This is not a stylistic preference. On a codebase with no tests, a suite written
by reading the implementation will faithfully encode its bugs and produce
confident green ticks over broken behaviour. Concretely:

- Prefer an **independent oracle** to an observation: a closed-form solution,
  `jax.grad` of a hand-written function, `optax` applied directly, the same
  computation in raw jax without pcx's wrapper.
- **Failing tests are the expected output of this stage,** not a problem to
  design around. A test that fails against the implementation is working.
- Where a test genuinely must record current behaviour — jax's traversal order,
  say — name it `test_characterises_*` and say in the docstring that it pins
  behaviour rather than asserting correctness.

## Scope

Modelled on how PyTorch or Lightning are tested: this is a library for training
neural networks, so the suite proves *the library* is correct. It does not
re-validate the science of predictive coding, and it does not assert end-to-end
research results. Numerical rigour is applied at component level — an energy
function, a gradient identity, an optimiser step — where a wrong answer is a
library bug rather than a research finding.

## Test tiers

### Tier 1 — Structural (`tests/core/`)

The pytree and parameter machinery. Pure CPU, milliseconds, no transforms.

| Area | Representative assertions |
| --- | --- |
| `Param` | flatten yields exactly one leaf; unflatten returns a new object of the same class; in-place `+=` preserves object identity; `bool(param)` raises |
| `StaticParam` | contributes zero dynamic leaves; `static(s) is s` is idempotent; mutation inside a trace does not escape |
| `BaseModule` | flatten/unflatten returns a new module holding *the same* `Param` objects — the property `shared()` is built on; attribute insertion order changes the treedef |
| `tree_ref` / `tree_unref` | round-trip preserves object identity for aliased params; leaf count contracts to the number of *unique* params; nesting round-trips; ref-index ordering is symmetric between the two functions |
| `tree_extract` / `tree_inject` | round-trip preserves values in order; `strict=True` raises on surplus |
| `RKG` | same seed gives the same stream; `split` advances state in place |

### Tier 2 — Transform equivalence (`tests/functional/`)

Every pcx transform is a wrapper around a jax primitive, which gives a free
oracle: **the wrapper must agree with hand-written raw jax.** This tier is where
the vmap defect surfaces.

- For `jit`, `vmap`, `value_and_grad`, `scan`, `while_loop`, `cond`, `switch`:
  compute a result through the pcx transform and through the equivalent raw jax
  expression, and assert they agree.
- Parameter write-back: mutating a `Param` passed as a **keyword** argument
  propagates to the caller's object; passing it **positionally** does not.
  This positional/keyword split is the library's central protocol.
- Parameter sharing: a param referenced twice inside one call is updated once,
  not twice, and its gradient is the sum of both paths (chain rule) rather than
  either contribution alone.
- Composition: `jit(vmap(...))` and `jit(value_and_grad(...))` agree with the
  unnested forms; only the outermost call refs and only the innermost unrefs.
- RNG: the global `RKG` key advances across a transform, and — the suspected
  defect — is left holding a concrete array rather than a tracer when the
  transformed function raises.

### Tier 3 — Component numerical correctness (`tests/numerics/`)

Where a wrong number is a library bug. Every expectation is a closed form or an
independent implementation.

- **Energy functions.** `se_energy == 0.5*(h-u)**2` against the closed form on
  random inputs — this catches a dropped ½, which silently halves the effective
  inference rate. `ce_energy` cross-checked against
  `optax.softmax_cross_entropy`, plus shift-invariance in `u` and no NaN at
  `u = 1e4`.
- **Gradient identities.** `∂/∂h se_energy == (h - u)` and
  `∂/∂u se_energy == -(h - u)` via `jax.grad`. This is *the* prediction-error
  identity: a sign flip here inverts inference while still appearing to train.
  For `ce_energy` with one-hot `h`, `∂/∂u == softmax(u) - h`.
- **Optimiser.** `Optim` wrapping `optax.sgd(lr)` with a constant gradient must
  give exactly `w - lr*g`; momentum checked over three steps against a
  hand-computed sequence. Gradients passed to `step` must not be mutated by
  `scale_by`. Masked-out parameters must not move; targeted ones must.
- **Vode.** The ruleset state machine: which rules fire on `set`/`get` for a
  given status, and that a `frozen` value is bit-identical after an inference
  step.
- **Layers.** `Layer(cls, ...)(x)` equals the bare equinox `cls(...)(x)` for the
  same key — the only thing that catches a parameter/static misclassification.
  `Dropout` is identity after `.eval()`; `BatchNorm` running stats change in
  train and not in eval.

### Tier 4 — Device smoke (`tests/devices/`, local only)

A small set of end-to-end training-loop runs — build a model, run inference and
weight steps, assert energy decreases and weights change — executed per backend:
CPU, CUDA, and Apple Metal via `jax-metal`.

These are **excluded from GitHub Actions**, which has no GPU. They are marked
`@pytest.mark.device` and skip automatically when the backend is unavailable, so
`just test-devices` is meaningful on a workstation and silent elsewhere.

This tier is deliberately thin. Its job is to catch total breakage on a backend,
not to localise faults — Tiers 1–3 do that.

## Handling the failures

Tests are written to assert correct behaviour and **allowed to fail**. They are
not `xfail`-ed, because `xfail` renders a real defect invisible in the ordinary
green run, which is the failure mode that produced this situation.

To keep CI a usable gate while the backlog is worked down, known-failing tests
carry a marker:

```python
@pytest.mark.bug("pxu.step does not restore status after the block; see BUGS.md#4")
def test_step_restores_status_after_block(): ...
```

The blocking CI job runs `-m "not bug"`. A second advisory job runs `-m bug` and
reports — the same pattern already used for `ty` in this repo, so there is one
convention rather than two. A marker is a claim about a specific defect and is
removed the moment that defect is fixed.

Confirmed defects are catalogued in `BUGS.md` with a reproduction, ranked by
severity, so fixing them is a separate reviewable pass rather than something
smuggled into the test PR.

## Fixtures and isolation

`RKG` is module-level global state seeded from the wall clock at import
(`pcx/core/_random.py`). Any test that constructs a layer or a Vode advances it,
which makes results order-dependent. An **autouse fixture reseeds `RKG` before
every test and restores its key afterwards**; without it the suite is not
reproducible. The existing `conftest.py` already pins the CPU backend before jax
is imported.

Markers: `bug` (known defect), `device` (needs a real accelerator), `slow`.
Default `addopts` excludes `device`.

## CI

| Job | Runs | Blocking |
| --- | --- | --- |
| Test matrix (3 OS × Python 3.11–3.13) | `-m "not (bug or device)"` | yes |
| Known bugs | `-m bug` | no, advisory |
| jax compatibility | oldest-supported and newest jax, Tiers 1–3 | yes |

The jax-compatibility leg is the direct answer to the vmap defect: resolving
against something other than the lockfile is the only way this class of bug is
ever caught.

## Confirmed defects (independently reproduced)

Ranked by severity. Each was reproduced directly, not inferred.

1. **`pxf.vmap` fails on jax >= 0.4.34** — `ValueError: pytree structure error
   ... tree_map tree[...]['__RKG']`. The library cannot run its own tutorials.
2. **`save_params`/`load_params` corrupts models with shared parameters** —
   duplicate references are written as `None`, stored as a `dtype=object` array,
   and `np.load` refuses it: `ValueError: Object arrays cannot be loaded when
   allow_pickle=False`. Checkpoints are unrecoverable.
3. **`pxu.step` does not restore status** — after
   `with pxu.step(model, STATUS.INIT):` the model is left in `'init'`, so any
   energy computed before the next `step` block runs under the wrong ruleset.
   The block also lacks `try/finally`, so an exception skips cleanup entirely.
4. **`Vode.energy()` raises after a cache clear** —
   `TypeError: argument of type 'NoneType' is not iterable`. `ParamDict` guards
   `__setitem__` against a `None` value but not `__contains__`, and
   `clear_params` produces exactly that state.
5. **`EnergyModule.energy()` raises on a model with no Vodes** —
   `functools.reduce` with no initial value. One-word fix.

Two further items are unverified and stay out of the plan until reproduced: a
suspected `~` negation bug in the mask DSL (**did not reproduce** — negation
selected a different set, as intended) and suspected double-counting of shared
submodules in `EnergyModule.energy`.

## Out of scope

- Fixing the defects above. Tests first, per the agreed sequencing.
- End-to-end validation of research results (PC-versus-backprop gradient
  equivalence, convergence on real datasets).
- Running the tutorial notebooks in CI. They cannot pass until defect 1 is fixed;
  revisit afterwards.

## Success criteria

High coverage is the destination, but it is **not** the target for this round,
and it is not how this round should be judged. Coverage is trivially gamed by
tests that execute code without asserting anything meaningful about it, and on a
codebase this size the temptation to pad is real. The first round is measured by
what it *finds*:

- Every confirmed defect has a test that fails for the right reason, and
  `BUGS.md` reproduces it.
- The key components are covered where a fault would be severe and silent: the
  ref/unref identity machinery, the transform write-back protocol, the energy
  and gradient identities, the optimiser arithmetic.
- Every expectation is traceable to an oracle rather than to observed output.
- The blocking suite runs in under 60 seconds on CPU.

Explicitly **not** a criterion for this round: a coverage percentage. Each test
must justify its existence by the class of bug it would catch — if it would not
fail for any plausible defect, it does not belong. Coverage gaps that remain
after this round are recorded rather than filled with filler, and later rounds
close them deliberately. Expect the first round to leave `pcx/utils/_data.py`
(a six-line stub) and much of the layer catalogue untouched; that is correct
prioritisation, not an omission.
