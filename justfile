set quiet

# ─── Default ──────────────────────────────────────────────────────────────────

# List all available recipes
default:
    @just --list --unsorted

# ─── Environment ──────────────────────────────────────────────────────────────

# Create the dev environment (library + dev tools)
install:
    uv sync --group dev

# Update the lockfile to the latest allowed versions
update:
    uv lock --upgrade

# ─── Code Quality ─────────────────────────────────────────────────────────────

# Format code with ruff
format:
    uv run ruff format .

# Check formatting without modifying files
format-check:
    uv run ruff format --check .

# Lint code with ruff
lint:
    uv run ruff check .

# Lint and auto-fix issues
lint-fix:
    uv run ruff check . --fix

# Run the ty type checker. Advisory: pcx carries a baseline of pre-existing
# diagnostics from its dynamic pytree design, so this reports without failing —
# matching the CI job. Use `typecheck-strict` to make it a gate.
typecheck:
    -uv run ty check

# Run ty as a hard gate (non-zero exit on any diagnostic)
typecheck-strict:
    uv run ty check

# Run every read-only quality gate
check: format-check lint typecheck

# Auto-fix lint issues, then format. Order matters: `ruff check --fix` can leave
# its rewrites unformatted, so formatting has to run last.
fix: lint-fix format

# ─── Testing ──────────────────────────────────────────────────────────────────

# Run the test suite (excludes known-bug and device tests)
test *ARGS:
    uv run pytest -v {{ ARGS }}

# Run the known-defect tests. These are EXPECTED TO FAIL — each asserts correct
# behaviour that a current bug violates. See BUGS.md.
test-bugs *ARGS:
    -uv run pytest -m bug -v {{ ARGS }}

# Run accelerator smoke tests locally. Backends that aren't present are skipped.
# JAX_PLATFORMS is cleared so jax can see the GPU/Metal devices.
test-devices *ARGS:
    JAX_PLATFORMS='' uv run pytest -m device -v {{ ARGS }}

# Everything: normal suite, then the bug catalogue, then device smokes
test-all *ARGS:
    uv run pytest -m "" -v {{ ARGS }}

# Run tests with coverage and print a report
coverage *ARGS:
    uv run coverage run -m pytest -v {{ ARGS }}
    uv run coverage report

# Generate an HTML coverage report and open it
coverage-html: coverage
    uv run coverage html
    @echo "Report written to htmlcov/index.html"

# ─── Docs ─────────────────────────────────────────────────────────────────────

# Build the Sphinx documentation into docs/_build/html
docs:
    mkdir -p docs/source/examples
    cp -r examples/* docs/source/examples/
    uv run --group docs sphinx-apidoc -f -o docs/source/ pcx/
    uv run --group docs sphinx-build -b html docs docs/_build/html
    @echo "Docs written to docs/_build/html/index.html"

# ─── Release ──────────────────────────────────────────────────────────────────

# Build the wheel and sdist into dist/
build:
    uv build

# Preview the release notes that a given version tag would publish
release-notes VERSION:
    uv run python .github/scripts/extract_changelog.py {{ VERSION }}

# ─── All-in-One ───────────────────────────────────────────────────────────────

# Fix, check and test — run this before opening a PR
all: fix check test

# Remove build artifacts and caches
clean:
    rm -rf .coverage .coverage.* htmlcov/ dist/ .pytest_cache/ .ruff_cache/ docs/_build/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Mutation testing: inject deliberate defects and check the suite catches them
mutation-test:
    uv run python scripts/mutation_test.py
