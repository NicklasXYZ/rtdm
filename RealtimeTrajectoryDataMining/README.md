# rtdm (realtime trajectory data mining()

A codebase containing different tools for processing realtime trajectory data.

## Overview

### Data

TODO

## Installation

``` bash
poetry install;
pre-commit install
```

## Usage

``` bash
# Install dependencies
poetry install

# Generate database tables
cd rtdm;
python manage.py make migrations && python manage.py migrate
```

See notebooks and examples in the `./scripts` directory.

## Development

``` bash
# Make sure pre-commit is installed:
pip install pre-commit

# Install pre-commit script
pre-commit install

# Run hooks on all files
pre-commit run --all-files

# ... or Run hooks on specific files
pre-commit run --files /RealtimeTrajectoryDataMining/*

# Build docs
cd docs;
make clean && make html
```