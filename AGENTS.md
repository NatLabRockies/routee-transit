# AGENTS.md - RouteE-Transit

This document provides guidance for AI coding assistants working on the RouteE-Transit project.

## Project Overview

**RouteE-Transit** (`nrel.routee.transit`) is a Python package that predicts energy consumption for transit bus trips based on GTFS (General Transit Feed Specification) data and RouteE-Powertrain models. The package:

- Matches GTFS shapes to the OpenStreetMap road network
- Aggregates speed, distance, and grade estimates at the OSM road link level
- Uses trained RouteE-Powertrain models to predict energy consumption
- Supports HVAC energy impacts for battery-electric buses
- Models both mid-block deadhead (between trips) and depot deadhead (pull-out/pull-in) trips

## Technology Stack

- **Language**: Python 3.10-3.12
- **Package Manager**: Pixi (recommended) or pip
- **Build System**: Hatch
- **Key Dependencies**:
  - `gtfsblocks` - GTFS data processing
  - `nrel-routee-powertrain` - Energy consumption modeling
  - `mappymatch` - Map matching algorithms
  - `geopandas` - Geospatial data handling
  - `gradeit` - Grade/elevation processing
  - `nrel-routee-compass` - Routing engine (local editable dependency)

## Project Structure

```
routee-transit/
├── nrel/routee/transit/     # Main package code
├── tests/                   # Test suite
├── scripts/                 # Example scripts and utilities
├── docs/                    # Documentation and examples
├── FTA_Depot/              # Default depot location data (NTD 2023)
├── TMY/                    # Typical Meteorological Year data
├── sample-inputs/          # Sample GTFS and configuration files
├── pyproject.toml          # Project configuration
└── pixi.lock               # Pixi lock file
```

## Development Workflow

### Environment Setup

**Using pip and conda**:
```bash
conda activate routee-transit
maturin develop
```

### Running Checks 

#### python 

Make sure to do `maturin develop` first if any rust code changed
```bash
ruff format
ruff check --fix
mypy .
pytest tests/
```

#### rust
```bash
cd rust
cargo test --workspace
cargo fmt --all --workspace
cargo clippy
cargo sort --check --workspace
```

## Key Concepts

### GTFS Data Processing

The package works with standard GTFS feeds, processing:
- `trips.txt` - Transit trips
- `stop_times.txt` - Stop sequences and times
- `shapes.txt` - Route geometries
- `stops.txt` - Stop locations
- `routes.txt` - Route information

### Deadhead Trips

**Depot Deadhead**: Trips between depots and the first/last stops of a block
- Pull-out: Depot → First stop
- Pull-in: Last stop → Depot
- Default depot locations from NTD 2023 dataset

**Mid-block Deadhead**: Trips between consecutive trips in a block when they don't share endpoints

### Map Matching

The package uses map matching to align GTFS shapes with the OpenStreetMap road network, enabling accurate speed, distance, and grade estimation.

### Energy Prediction

Uses RouteE-Compass to predict energy consumption based on:
- Route characteristics (distance, speed, grade)
- Vehicle type (e.g., battery-electric buses)
- Environmental factors (HVAC loads, ambient temperature)

## Common Development Tasks

### Adding New Features

1. **Understand the pipeline**: The main entry point is `GTFSEnergyPredictor.run()` which orchestrates the entire workflow
2. **Identify the component**: Determine which module your feature affects (e.g., deadhead routing, map matching, energy prediction)
3. **Write tests first**: Add tests in `tests/` directory
4. **Implement the feature**: Add code to appropriate module in `nrel/routee/transit/`
5. **Update documentation**: Add examples to `docs/examples/` if user-facing

### Debugging Issues

1. **Check GTFS data quality**: Many issues stem from malformed or incomplete GTFS feeds
2. **Verify depot locations**: Ensure depot shapefile exists and is readable
3. **Review map matching results**: Map matching can fail if GTFS shapes are far from OSM network

### Working with Geospatial Data

- Always use `EPSG:4326` (WGS84) for lat/lon coordinates
- Use `EPSG:3857` (Web Mercator) for distance calculations
- Ensure CRS consistency when merging GeoDataFrames
- Use `geopandas` for spatial operations

### Type Hints

- Use strict type hints (mypy strict mode is enabled)
- Import types from `typing` module
- Use `Any` sparingly and only when necessary
- Document complex types in docstrings

## Testing Guidelines

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test complete workflows
- **Use pytest fixtures**: For common test data and setup
- **Test with real GTFS data**: Use sample feeds from `sample-inputs/`
- **Mock external dependencies**: When testing RouteE-Compass or RouteE-Powertrain interactions

## Documentation

- **Docstrings**: Use NumPy-style docstrings for all public functions and classes
- **Examples**: Add working examples to `docs/examples/`
- **Building docs**: Run `pixi run docs` to build with jupyter-book
- **API reference**: Auto-generated from docstrings

## Common Pitfalls

3. **Time Format**: GTFS times can exceed 24 hours (e.g., "25:30:00" for 1:30 AM next day)
4. **Block IDs**: Not all GTFS feeds have block IDs; handle `NaN` values appropriately
5. **CRS Mismatches**: Always verify coordinate reference systems when working with geospatial data

## Questions to Ask

When working on this codebase, consider:

1. **Does this change affect the public API?** If so, update examples and documentation
4. **Is this change backward compatible?** Consider existing users and workflows
5. **Are units consistent?** Verify distance (meters), time (seconds), energy (kWh) units

## Resources

- **GTFS Specification**: https://gtfs.org/
- **NTD Depot Data**: https://data.transportation.gov/stories/s/gd62-jzra
- **Project Documentation**: https://natlabrockies.github.io/routee-transit/
- **RouteE-Powertrain**: https://github.com/NatLabRockies/routee-powertrain
- **RouteE-Compass**: https://github.com/NatLabRockies/routee-compass

## Contact & Support

This is an NLR (National Laboratory of the Rockies) project. When making significant changes, consider:
- Impact on research reproducibility
- Compatibility with existing workflows
- Documentation for scientific users
