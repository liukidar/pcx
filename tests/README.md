# Testing strategy

## The rule

**Derive every expectation from what the code is meant to do, never from what it
currently returns.** Write the closed form, or the raw-jax equivalent, or drive
`optax` directly, and compare against that. A test written by observing the
implementation faithfully encodes its bugs and hands you a green tick over broken
behaviour. If a test must pin current behaviour instead, name it
`test_characterises_*` and say so in the docstring.

## Tiers

Each tier is defined by the oracle it checks against, not by the code it touches.

| Directory | What it proves | Oracle |
| --- | --- | --- |
| `core/` | pytree and parameter machinery | structural invariants |
| `functional/` | the jax-wrapper transforms | hand-written raw jax |
| `numerics/` | energies, gradients, optimiser, layers | closed forms, `optax`, bare `equinox` |
| `nn/`, `utils/` | layers, masks, serialisation, `step` | documented contracts |
| `devices/` | one training step per backend | local only, never CI |

`test_imports.py` and `test_smoke.py` guard the package surface: every public name
imports, and a two-layer model trains end to end.

## Markers

```shell
just test           # default gate: excludes bug and device
just test-bugs      # the known-defect catalogue, expected to fail
just test-devices   # accelerator smokes; absent backends skip
just test-all       # everything
```

**`bug`** asserts correct behaviour that a known defect currently violates.
Deliberately not `xfail`-ed: `xfail` hides a real defect inside a green run, which
is how these survived in the first place.

Every `bug` marker opens with the tracker reference that owns the defect, so the
marker is the map from test to fix:

```python
@pytest.mark.bug("#70: Param defines __idiv__ (Python 2) but no __itruediv__, so `p /= x` rebinds")
```

`#N` is a GitHub issue. `BUGS.md#N` is a defect still under discussion; that file is
deleted once they are all resolved. **To fix a defect: grep its reference, delete
those markers, and the tests become its regression guard.**

**`device`** needs a real accelerator. GitHub Actions has no GPU, so these never run
there. Run `just test-devices` locally if you touch a GPU code path.

**`slow`** takes more than a couple of seconds.

## Fixtures

`conftest.py` pins `JAX_PLATFORMS=cpu` before jax is imported, and an autouse fixture
reseeds `pcx.RKG` around every test. `RKG` is wall-clock-seeded module-level state and
the default argument of every layer and Vode constructor, so without that reseed the
suite is order-dependent. Use the `key` and `rkg` fixtures for anything random, and
`assert_allclose` / `tree_allclose` / `count_leaves` for comparisons.

## Does a passing test mean anything?

`just mutation-test` injects ten deliberate one-line defects into the library and
checks the suite notices. All ten are caught, each by a specifically named assertion
rather than an incidental error downstream. Run it when you are unsure whether a new
test would actually catch anything.

One caveat it taught us: a surviving mutant is not automatically a gap. Flipping
`se_energy` to `u - h` survives because `0.5(u-h)² == 0.5(h-u)²` exactly and the
gradient is unchanged. That is an equivalent mutant, not a blind spot.
