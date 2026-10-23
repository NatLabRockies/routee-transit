# Installation

## Prerequisites

- Python version >=3.10, <3.13
- Git
- [Rust](https://www.rust-lang.org/tools/install) (required to build RouteE-Transit's
  Rust extension, which is compiled from source at install time)

## Installation with pip

### 1. Clone the repository

```bash
git clone https://github.com/NatLabRockies/routee-transit.git
cd routee-transit
```

### 2. Create a virtual environment (recommended)
e.g., using `conda`:

```bash
conda create -n routee-transit python=3.12
conda activate routee-transit
```

### 3. Install the package
From the root directory,
```bash
pip install .
```

```{note}
RouteE-Transit depends on `geopandas` and `osmnx`, which in turn need GDAL. If the
install fails with a GDAL-related error on macOS, run `brew install gdal` and try again.
Installing with Pixi (below) avoids this by managing GDAL for you.
```

## Installation with Pixi

[Pixi](https://pixi.sh/) handles both the Python and system (GDAL, Rust) dependencies,
and is the recommended path for development:

```bash
git clone https://github.com/NatLabRockies/routee-transit.git
cd routee-transit
pixi install
pixi shell -e dev-py312
```

## Setup for developers
See [](contributing)

For a development installation with all optional dependencies:
```bash
pip install -e ".[dev]"
```

