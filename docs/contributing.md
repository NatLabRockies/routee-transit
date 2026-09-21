# Developer Guide

## Set Up Development Environment with Pixi

[Pixi](https://pixi.sh/) is a modern package manager that handles both Python and system dependencies automatically.

The `routee-transit` `pyproject.toml` file defines various development environments with Pixi for different Python versions (e.g. `dev-py310`, `dev-py311`, etc.) to ease development and testing. Follow the instructions below to configure your environment with Pixi.

### 1. Install pixi

Follow the installation instructions at [pixi.sh](https://pixi.sh/latest/#installation) or use:

```bash
# On macOS/Linux
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Clone and set up the project

```bash
git clone https://github.com/NatLabRockies/routee-transit.git
cd routee-transit
pixi install
```

### 3. Activate the environment

```bash
pixi shell
```

To activate a specific environment (not just the default), use the `-e` flag:

```bash
pixi shell -e dev-py310
```

Alternatively, you can use `pixi run -e dev-py310 python myfile.py` to execute a python file in the environment specified. If you're using VS Code, the [Pixi VSCode](https://marketplace.visualstudio.com/items?itemName=jjjermiah.pixi-vscode) extension is useful.

## Building the Rust Extension

Part of RouteE-Transit is implemented in Rust (see the `rust/` directory) and is built
with [maturin](https://www.maturin.rs/). Pixi builds the extension for you as part of
`pixi install`, but after changing any Rust code you must rebuild before the change
takes effect in Python:

```bash
maturin develop
```

## Running Checks

The Python checks that CI runs are available as pixi tasks:

```bash
pixi run -e dev-py312 fmt_fix     # ruff format
pixi run -e dev-py312 lint_fix    # ruff check --fix
pixi run -e dev-py312 typing      # mypy .
pixi run -e dev-py312 test        # pytest tests/
pixi run -e dev-py312 check       # all of the above
```

CI runs this matrix against Python 3.10, 3.11, and 3.12, so verify against each
`dev-py310` / `dev-py311` / `dev-py312` environment before opening a pull request.
mypy runs in strict mode.

For the Rust crates:

```bash
cd rust
cargo test --workspace
cargo fmt --all
cargo clippy
```

## Build Documentation
To build the documentation locally with `jupyter-book`, use the pixi task defined in `pyproject.toml`:

```bash
pixi run docs
```

Be aware that `jupyter-book` will run all of the documentation examples when the docs are built, which can take a few minutes. To build the documentation faster, you can skip the examples by simply commenting them out in the table of contents (`docs/_toc.yml`).
