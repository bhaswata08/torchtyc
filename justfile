# torchtyc development tasks. Run `just` to see them.

# List the available recipes.
default:
    @just --list

# Install every dependency, including the extras and the dev tools.
sync:
    uv sync --all-extras --dev

# Run the test suite. Extra arguments go through to pytest.
test *args:
    uv run pytest {{args}}

# Lint and check formatting, over the same paths CI checks.
lint:
    uv run ruff check src tests bench
    uv run ruff format --check src tests bench

# Reformat, and apply the lint fixes ruff can make itself.
fmt:
    uv run ruff format src tests bench
    uv run ruff check --fix src tests bench

# Run torchtyc over its own deliberately-wrong fixtures, where findings are the point.
self-check:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run torchtyc check tests/fixtures --format github
    status=$?
    if [ "$status" -gt 1 ]; then
        echo "torchtyc exited $status, which is torchtyc failing rather than a finding" >&2
        exit "$status"
    fi

# Everything CI runs, in the order CI runs it.
ci: lint test self-check

# Print the current version.
version:
    @sed -n 's/^__version__ = "\(.*\)"/\1/p' src/torchtyc/__init__.py

# Set the version in src/torchtyc/__init__.py, which pyproject.toml reads from.
bump new:
    #!/usr/bin/env bash
    set -euo pipefail
    grep -q '^__version__ = ' src/torchtyc/__init__.py
    sed -i 's/^__version__ = .*/__version__ = "{{new}}"/' src/torchtyc/__init__.py
    echo "version is now $(just version)"

# Build the wheel and the sdist into dist/.
build:
    rm -rf dist
    uv build

# Build and check what would be published, without publishing it.
release-dry new: (bump new) ci build
    @echo
    @ls -l dist
    @echo
    @echo "nothing published. src/torchtyc/__init__.py is left at {{new}}."

# Cut a release: gate, bump, verify, build, publish to PyPI, tag, push.
release new:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ -n "$(git status --porcelain)" ]; then
        echo "the working tree is dirty; commit or stash first" >&2
        exit 1
    fi
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
        echo "releases are cut from main" >&2
        exit 1
    fi
    git fetch --quiet origin main
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
        echo "HEAD and origin/main disagree; push or pull first" >&2
        exit 1
    fi
    if git rev-parse -q --verify "refs/tags/v{{new}}" >/dev/null; then
        echo "tag v{{new}} already exists" >&2
        exit 1
    fi
    if command -v gh >/dev/null 2>&1; then
        conclusion=$(gh run list --branch main --limit 1 --json conclusion --jq '.[0].conclusion')
        if [ "$conclusion" != "success" ]; then
            echo "CI on main is '$conclusion', not success" >&2
            exit 1
        fi
    else
        echo "gh is not installed, so CI status was not checked" >&2
    fi
    if [ ! -f .env ]; then
        echo ".env is missing, so UV_PUBLISH_TOKEN is unavailable" >&2
        exit 1
    fi
    if [ ! -t 0 ] && [ "${CONFIRM:-}" != "yes" ]; then
        echo "stdin is not a terminal, so the publish cannot be confirmed." >&2
        echo "run this from a terminal, or pass CONFIRM=yes to answer in advance." >&2
        exit 1
    fi

    just bump {{new}}
    just ci
    just build

    echo
    ls -l dist
    echo
    if [ "${CONFIRM:-}" = "yes" ]; then
        reply=y
    else
        read -r -p "publish torchtyc {{new}} to PyPI? a version cannot be reused [y/N] " reply || reply=""
    fi
    if [ "$reply" != "y" ]; then
        git checkout src/torchtyc/__init__.py
        echo "aborted, and the bump is reverted; nothing was committed"
        exit 1
    fi

    git commit -am "release: {{new}}"

    set -a
    . ./.env
    set +a
    uv publish

    git tag -a "v{{new}}" -m "torchtyc {{new}}"
    git push origin main "v{{new}}"
    echo
    echo "torchtyc {{new}} is on PyPI and v{{new}} is pushed."

# Remove the build and cache directories.
clean:
    rm -rf dist build .pytest_cache .ruff_cache
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
