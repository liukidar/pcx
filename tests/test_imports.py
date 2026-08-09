"""Every subpackage must import.

`pcx/__init__.py` only re-exports `pcx.core`, so `import pcx` succeeds even when a
sibling subpackage is broken at module level. Annotations are the usual culprit: a
forward reference in an evaluated union (`"Foo" | Callable`) raises at import time
unless the whole annotation is quoted.
"""

import importlib

import pytest

SUBPACKAGES = [
    "pcx",
    "pcx.core",
    "pcx.functional",
    "pcx.nn",
    "pcx.predictive_coding",
    "pcx.utils",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str):
    importlib.import_module(name)


def test_documented_entry_points_are_reachable():
    """The names the tutorials use, under the aliases the tutorials use."""
    import pcx.functional as pxf
    import pcx.nn as pxnn
    import pcx.predictive_coding as pxc
    import pcx.utils as pxu

    for module, names in (
        (pxf, ["jit", "vmap", "value_and_grad", "scan", "while_loop", "cond"]),
        (pxu, ["Optim", "M", "step"]),
        (pxc, ["Vode", "EnergyModule", "se_energy", "ce_energy"]),
        (pxnn, ["Linear", "Conv2d", "LayerParam"]),
    ):
        for name in names:
            assert hasattr(module, name), f"{module.__name__}.{name} is missing"
