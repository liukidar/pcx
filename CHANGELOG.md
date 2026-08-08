# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
How to release:
  1. Move the entries below out of [Unreleased] into a new `## [X.Y.Z] - YYYY-MM-DD` section.
  2. Bump `version` in pyproject.toml to the same X.Y.Z.
  3. git tag vX.Y.Z && git push origin vX.Y.Z
The Release workflow refuses to publish if the tag, pyproject version and a
matching changelog section do not all agree, and it uses that section as the
GitHub release notes.
-->

## [Unreleased]

### Added

- Test suite of 442 tests across four tiers: structural (`tests/core/`), transform equivalence against hand-written raw jax (`tests/functional/`), numerical correctness against closed forms and `optax` (`tests/numerics/`), plus `tests/utils/`, `tests/nn/` and per-backend smoke tests (`tests/devices/`). Coverage of `pcx/` is 96%.
- 29 verified defects, each with a failing test asserting the correct behaviour. Marked `bug` and excluded from the default run; reported by an advisory CI job. Every marker opens with the issue that owns the defect, so the marker is the map from test to fix. 19 are tracked as issues #67 to #79; the 10 still under discussion with the maintainers live in `BUGS.md`, which is deleted once they are resolved.
- `tests/README.md` documenting the testing strategy: the tiers and their oracles, the `bug` / `device` / `slow` markers, the shared fixtures, and how mutation testing validates the suite.
- `scripts/mutation_test.py`, which injects deliberate defects to confirm the suite detects them. It currently catches 10 of 10.
- Autouse fixture reseeding the global `pcx.RKG` around every test, since it is wall-clock-seeded module state and the default argument of every layer and Vode constructor, so without it the suite is order-dependent.
- `device` marker and `just test-devices` for accelerator smoke tests (CPU, CUDA, Apple Metal), which skip when a backend is absent and never run in CI.
- `uv` as the package and environment manager, replacing Poetry. Dependencies are locked in `uv.lock`.
- `ty` as the type checker, wired into CI as an advisory (non-blocking) job.
- `pytest` test suite with a `tests/` package, coverage configuration and Codecov reporting.
- `justfile` with recipes for the full developer loop (`just install`, `just check`, `just test`, `just all`).
- `CHANGELOG.md` plus `.github/scripts/extract_changelog.py`, which turns the section for a version into GitHub release notes.
- CI test matrix across Linux, macOS and Windows on Python 3.11–3.13.
- A build job that verifies the wheel installs and imports on a clean interpreter.
- Dependabot updates for GitHub Actions, so retired action versions are caught before they break a workflow.
- `codecov.yml` configuring informational coverage status checks while the suite is built out.

### Changed

- `pyproject.toml` migrated from the Poetry table layout to PEP 621 `[project]` metadata with `hatchling` as the build backend.
- Ruff now runs against an explicit rule selection (`E`, `W`, `F`, `I`, `UP`, `B`, `C4`, `RUF`) so linting does not drift between ruff releases.
- The five separate workflows (format, lint, type-check, test, publish) are consolidated into `ci.yml` and `release.yml`. The old workflows referenced a `pcax` directory that no longer exists, and the test and type-check workflows were entirely commented out.
- Release publishing is now tag-driven and verifies that the git tag, `pyproject.toml` version and `CHANGELOG.md` section all agree before uploading.
- The Docker image and dev container install with `uv` rather than Poetry, and both now build a `just` toolchain.
- `.readthedocs.yaml` moved from `docs/` to the repository root and installs the `docs` dependency group with `uv`.
- `docs/source/` is now generated at build time rather than committed. The tutorial notebooks are mirrored in from `examples/` and the API stubs come from `sphinx-apidoc`; `just docs` runs the same steps locally.

### Removed

- The committed `docs/source/pcax.*.rst` API stubs, which documented a `pcax` package that no longer exists — so the published API reference had been empty.

### Fixed

- `just fix` ran the formatter before the linter, so `ruff check --fix` could leave its rewrites unformatted and `just all` would then fail its own format check.
- Various lint findings across `pcx`: unsorted imports, deprecated `typing` aliases, implicit `Optional` annotations, stale `# noqa` codes, and an unnecessary `map`/`zip` pairing in `BaseModule.flatten_module_with_keys`.

## [0.6.2.post3] - 2024-11-03

Releases up to and including 0.6.2.post3 predate this changelog. See the
[GitHub releases](https://github.com/liukidar/pcx/releases) for their history.
