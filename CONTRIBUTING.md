# Contributing to PCX

## Setup

PCX uses [uv](https://docs.astral.sh/uv/) for packaging and [just](https://github.com/casey/just) as the task runner.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
git clone https://github.com/liukidar/pcx.git
cd pcx
just install
```

`just install` creates a `.venv` from `uv.lock`, so everyone gets byte-identical dependencies. Run `just` on its own to list every recipe.

## The loop

Work on a branch, and before opening a pull request run:

```shell
just all     # format, auto-fix, check, test
```

That is the same set of gates CI runs, so a green `just all` should mean a green pipeline.

| Gate        | Command           | Tool                                            | Blocking |
| ----------- | ----------------- | ----------------------------------------------- | -------- |
| Format      | `just format`     | [ruff format](https://docs.astral.sh/ruff/formatter/) | yes  |
| Lint        | `just lint`       | [ruff](https://docs.astral.sh/ruff/linter/)     | yes      |
| Type check  | `just typecheck`  | [ty](https://github.com/astral-sh/ty)           | no       |
| Tests       | `just test`       | [pytest](https://docs.pytest.org/)              | yes      |

Ruff's rule selection lives in `pyproject.toml` under `[tool.ruff.lint]`, pinned explicitly so linting does not shift underneath you when ruff releases a new version.

Type checking is **advisory for now**. `pcx` carries a baseline of roughly 115 `ty` diagnostics that come from its dynamic pytree design — modules that rewrite `__dict__` during unflattening, and parameters that forward attribute access to the array they wrap. The CI job reports them but does not fail the build. Please do not add new ones; once the baseline is cleared, the job becomes a hard gate.

## Dependencies

Add packages with `uv`, never by hand-editing the lockfile:

```shell
uv add equinox            # runtime dependency
uv add --group dev pytest # development tooling
uv lock --upgrade         # refresh the lock to the newest allowed versions
```

Commit `pyproject.toml` and `uv.lock` together. A `pyproject.toml` that has drifted from `uv.lock` breaks environment setup for everyone.

## Tests

Tests live in `tests/` and run against the CPU backend — `tests/conftest.py` pins `JAX_PLATFORMS=cpu` before JAX is imported, so results do not depend on whether a GPU is present. Use the `key` fixture for anything that draws randomness, so failures are reproducible.

Add tests for any behaviour you add or change. CI runs the suite on Linux, macOS and Windows across Python 3.11–3.14.

GPU code paths cannot be exercised on GitHub Actions. If your change touches them, test on a GPU machine locally and say so in the pull request.

## Changelog

Every user-visible change gets an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), in the appropriate `### Added` / `### Changed` / `### Fixed` / `### Removed` subsection. Reference the PR number.

Purely internal changes — a refactor with no behavioural effect, a typo in a comment — do not need one.

## Releasing

1. Move the `[Unreleased]` entries into a new `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md`.
2. Bump `version` in `pyproject.toml` to the same `X.Y.Z`.
3. Merge, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

The `Release` workflow refuses to publish unless the tag, the `pyproject.toml` version and a matching changelog section all agree. It then builds with `uv build`, verifies the wheel imports on a clean interpreter, publishes to PyPI through [trusted publishing](https://docs.pypi.org/trusted-publishers/), and opens a GitHub release whose notes are that changelog section.

## Docker and Dev Containers

A pre-configured container is the least fiddly way to get a GPU environment. The `Dockerfile` lives in [docker/](docker/), with a `run.sh` that builds and runs it.

**The image needs CUDA 12.2 or later.** Check `nvidia-smi`; if it reports less, update the base `nvidia/cuda` image in [docker/Dockerfile](docker/Dockerfile) to match your host.

You need a CUDA >= 12.2 machine with an NVIDIA GPU (without one, skip the GPU passthrough), [Docker](https://docs.docker.com/engine/install/) newer than 20.10.9, [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) so Docker can reach the GPU (restart the daemon afterwards, `sudo systemctl restart docker` on Ubuntu), and [VS Code](https://code.visualstudio.com/download) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Open the project in VS Code and run `Dev Containers: Reopen in Container` (Ctrl/Cmd+Shift+P). The first build takes 15 to 30 minutes. `Dev Containers: Reopen folder locally` exits and `Dev Containers: Rebuild Container` rebuilds. `hostname` tells you where you are: 12 meaningless characters means you are inside the container.

The interpreter comes from uv, which provisions the version named in `.python-version`; the base image's own Python is older than this project supports. Add packages with `uv add <package>`, or `uv add --group dev <package>` for tooling.

## Pull requests

Add docstrings and comments to what you write. Request Luca as a reviewer, and once approved use **Squash and Merge** to keep the history tidy.

To skip CI on an intermediate commit, start the commit message with `[skip ci]`.
