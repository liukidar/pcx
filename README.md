<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/pcx-banner-dark-rounded.svg">
    <img src="docs/_static/pcx-banner-light-rounded.svg" alt="PCX — predictive coding in JAX" width="720">
  </picture>
</p>

# PCX — Predictive Coding Networks Made Simple

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/pcx.svg)](https://badge.fury.io/py/pcx)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://pcx.readthedocs.io/en/latest/)
[![CI](https://github.com/liukidar/pcx/actions/workflows/ci.yml/badge.svg)](https://github.com/liukidar/pcx/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/liukidar/pcx/graph/badge.svg)](https://codecov.io/gh/liukidar/pcx)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2407.01163-b31b1b.svg)](https://arxiv.org/abs/2407.01163)

A JAX library for building highly configurable predictive coding networks.

- **[Tutorials](examples/)** — nine notebooks, start with [`0_two_moons`](examples/0_two_moons.ipynb)
- **[Documentation](https://pcx.readthedocs.io/en/latest/)** — API reference and guides
- **[Benchmarking paper](https://arxiv.org/abs/2407.01163)** and the [code for its experiments](https://github.com/liukidar/pcax/releases/tag/v0.6.1)
- **[Research notes](notes.pdf)** — open questions Luca never had time to chase, summarised [below](#open-questions)
- **[Contributing](CONTRIBUTING.md)** — dev setup, tests, release process

## Installation

PCX needs Python 3.11 or newer.

```shell
pip install pcx                 # CPU
pip install "pcx[cuda12]"       # NVIDIA GPU on Linux, pulls the CUDA build of JAX
```

### Platform support

PCX is pure Python and ships a single universal wheel, so it installs anywhere Python does. Which accelerators you can actually use is decided by JAX rather than by PCX, so see [JAX's supported platforms](https://docs.jax.dev/en/latest/installation.html#supported-platforms). The one that catches people out: there are no CUDA wheels for native Windows, so GPU work there needs WSL2.

CI runs the test suite on Linux, macOS and Windows, across Python 3.11 to 3.14 and three JAX versions.

### Installing from source

To work on PCX itself, or to track `main`:

```shell
git clone https://github.com/liukidar/pcx.git
cd pcx
uv sync --group dev
```

That creates a `.venv` from the locked dependency set in `uv.lock`, so the environment is reproducible across machines. If you do not have [uv](https://docs.astral.sh/uv/) yet, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

Prefer a plain editable install? `pip install -e .` also works.

## Quick start

One training step on a two-layer network: the output node is clamped to the target, hidden nodes relax, then the weights update.

```python
import jax, jax.numpy as jnp, optax
import pcx.functional as pxf, pcx.nn as pxnn, pcx.predictive_coding as pxc, pcx.utils as pxu


class Model(pxc.EnergyModule):
    def __init__(self, dims):
        super().__init__()
        self.layers = [pxnn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])]
        self.vodes = [pxc.Vode() for _ in self.layers]
        self.vodes[-1].h.frozen = True  # clamp the output to the target

    def __call__(self, x, y=None):
        for layer, vode in zip(self.layers, self.vodes):
            x = vode(jax.nn.tanh(layer(x)))
        if y is not None:
            self.vodes[-1].set("h", y)
        return self.vodes[-1].get("u")


mask = pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0))


@pxf.vmap(mask, in_axes=(0, 0), out_axes=0)
def forward(x, y, *, model):
    return model(x, y)


@pxf.vmap(mask, in_axes=(0,), out_axes=(None, 0), axis_name="b")
def energy(x, *, model):
    y_ = model(x, None)
    return jax.lax.pmean(model.energy().sum(), "b"), y_


model = Model([2, 16, 2])
x, y = jnp.zeros((8, 2)), jnp.ones((8, 2))

with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    forward(x, y, model=model)  # forward-initialise the value nodes

optim = pxu.Optim(lambda: optax.adamw(1e-3), pxu.M(pxnn.LayerParam)(model))

with pxu.step(model, clear_params=pxc.VodeParam.Cache):
    (e, _), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to((False, True)), has_aux=True)(energy)(x, model=model)

optim.step(model, g["model"])
```

The [tutorials](examples/) build this up properly, and cover randomness, control flow, convolutional models and Z-IL.

## Development

```shell
just install    # create the dev environment
just all        # fix, check and test, run this before opening a PR
just            # list every recipe
```

The toolchain is [uv](https://docs.astral.sh/uv/), [ruff](https://docs.astral.sh/ruff/), [ty](https://github.com/astral-sh/ty) and [pytest](https://docs.pytest.org/). See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, container setup and release process.

## Documentation

The documentation is available at [pcx.readthedocs.io](https://pcx.readthedocs.io/en/latest/). To build it yourself, see [docs/README.md](docs/README.md) or run `just docs`.

## Open questions

From Luca, alongside the [research notes](notes.pdf):

> I've uploaded some old research [notes](https://github.com/liukidar/pcx/blob/main/notes.pdf) I never had time to dive deeper into. I'm not sure if they are still relevant, but if anyone finds any of it interesting, I am always happy to chat about it.
> In particular:
> - the weights initialisation may not be generating "good" gradients according to the xavier initialisation paper formulae, when used for PC networks (until page 9);
> - rec-lra (https://arxiv.org/abs/2002.03911) does something that the authors don't make explicit in the paper that maybe can be mathematically formalised and generalised to be applied to PC as well in order to create more interconnected networks (that propagate the energy faster) (page 9-10);
> - It could be that waiting for the network to converge during inference is actually wrong with the current formulation. This would explain a lot of the behvaiours/tricks we have experineced to make PCNs train effectively. However it is a big problem for PC since its theoretical formulation is based around the idea of state convergence via inference (page 11-12, sorry if it's a bit messy).

## Citation

If this library was useful in your work, please cite [our paper](https://arxiv.org/abs/2407.01163):

```bibtex
@article{pinchetti2024benchmarkingpredictivecodingnetworks,
      title={Benchmarking Predictive Coding Networks -- Made Simple},
      author={Luca Pinchetti and Chang Qi and Oleh Lokshyn and Gaspard Olivers and Cornelius Emde and Mufeng Tang and Amine M'Charrak and Simon Frieder and Bayar Menzat and Rafal Bogacz and Thomas Lukasiewicz and Tommaso Salvatori},
      year={2024},
      eprint={2407.01163},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01163},
}
```

For the code behind the experiments in that paper, see the [benchmark paper release](https://github.com/liukidar/pcax/releases/tag/v0.6.1).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md), and record user-visible changes in [CHANGELOG.md](CHANGELOG.md).
