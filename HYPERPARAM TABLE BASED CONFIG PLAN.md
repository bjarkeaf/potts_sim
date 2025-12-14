# Hyperparam Table-Based Config Plan

## Overview

Enable configuration files to load optimized hyperparameters from CSV tables while allowing easy modification of global simulation parameters.

## Current Workflow

`hpc/save_best_hyperparams.py` does:

1. Best hyperparameters (optimized for success rate, mean gap, or min gap) are extracted from results parquet files

2. `hpc/save_best_hyperparams.py` saves these hyperparameters to a CSV file, unique for each model-graph combination

## Proposed Feature

### Goal
Create a configuration system that references hyperparameter CSV files, allowing users to:
- Load pre-optimized hyperparameters from saved CSV tables
- Override global simulation parameters (e.g., `T`, `dt`, `num_runs`) without modifying individual hyperparameters
- Maintain model-specific and graph-specific hyperparameter combinations

### Implementation Design

**New YAML Parameter:**
```yaml
hyperparam_table: "path/to/best_hyperparams.csv"
```

**Behavior:**
1. Config file specifies baseline parameters (simulation time, time step, annealing schedules, etc.)
2. When `hyperparam_table` is set, the system:
   - Loads hyperparameters from the CSV for each model-graph combination
   - Uses CSV hyperparameters to override any conflicting parameters in the config
   - Applies global config parameters that aren't hyperparameters (e.g., `num_runs`, `T`, `dt`)

**Precedence Order:**
1. CSV hyperparameters (highest priority for model-specific params)
2. Config file baseline parameters
3. Default values

### Benefits
- Quickly re-run experiments with optimized hyperparameters under different conditions
- Easy iteration on non-hyperparameter settings without manual CSV editing
- Cleaner separation between optimization results and experimental configuration