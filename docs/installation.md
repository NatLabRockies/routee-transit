# Installation

## Prerequisites

- Python 3.9 or higher
- Git

## Installation with pip

### 1. Clone the repository

```bash
git clone https://github.com/NREL/routee-transit.git
cd routee-transit
```

### 2. Create a virtual environment (recommended)
e.g., using `conda`:

```bash
conda create -n routee-transit
conda activate routee-transit
```

### 3. Install the package
From the root directory,
```bash
pip install .
```

## Setup for developers
See [](contributing)

For development installation with all dependencies:
```bash
pip install -e ".[dev]"
```

