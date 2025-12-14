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

---

## Implementation Details

### Status: ✅ COMPLETED

The hyperparam table-based config feature has been fully implemented and tested.

### Files Modified

1. **`hpc/run_potts_sweep.py`** (~200 lines added)
   - Added `load_hyperparams_from_csv()` function (lines 172-217)
   - Added `parse_param_id()` function (lines 219-311)
   - Modified `generate_param_sets()` to accept `hyperparams_override` parameter (line 347-365)
   - Updated main() to load hyperparam table from config (lines 1442-1459)
   - Updated task generation loop to use hyperparam overrides (lines 1474-1509, 1635-1676)
   - Added hyperparam_table to broadcast_data for MPI distribution (line 1602)

2. **`hpc/configs/0_hyperparam_table_test.yaml`** (new file)
   - Test configuration demonstrating the feature
   - Uses existing CSV from `best_hyperparams/250623_prelim_max-3-cut_mean_gap.csv`

### How It Works

1. **Configuration**
   Add `hyperparam_table` parameter to YAML config:
   ```yaml
   hyperparam_table: "../best_hyperparams/250623_prelim_max-3-cut_mean_gap.csv"
   ```

2. **CSV Loading**
   - On startup, `load_hyperparams_from_csv()` reads the CSV file
   - Creates a dictionary mapping `(model, graph)` → hyperparameters
   - Validates CSV structure and model types

3. **Param ID Parsing**
   - `parse_param_id()` extracts hyperparameters from CSV param_id strings
   - Handles all model types: POLYNOMIAL, NEC, SIGMOID, CIM, QPDC, FIXED_AMPLITUDE
   - Supports schedules, factors, and linked/prototype parameters
   - Examples:
     - `"po3_blin(1/mu_max, 1/mu_max + 2)_gf0.8"` → `{poly_order: 3, beta_expr: "lin(1/mu_max, 1/mu_max + 2)", gamma_factor: 0.8}`
     - `"ar1.00e-02_rt2.1_ia-mu_max_gpf8.0"` → `{alpha_rate: 0.01, r_target: 2.1, initial_alpha: "-mu_max", gamma_factor: 8.0, gamma_is_prototype: True}`

4. **Parameter Override Logic**
   - When generating tasks, check if `(model_name, graph_name)` exists in hyperparam_table
   - If found, pass hyperparameters to `generate_param_sets()` as override
   - If override provided, function returns single param_set instead of sweep
   - Global parameters (`T`, `dt`, `num_runs`, `num_states`, `noise_factor`) always come from YAML

5. **MPI Distribution**
   - Hyperparam table is broadcast to all ranks
   - Each rank independently regenerates its assigned tasks using the same logic
   - Ensures deterministic task distribution across ranks

### Testing

Successfully tested with:
- Single graph (G1): 60 tasks (6 models × 10 runs)
- Multiple graphs (G1, G2): 120 tasks (6 models × 2 graphs × 10 runs)
- All hyperparameters correctly loaded from CSV
- Wall time estimation working correctly

### Usage Example

```bash
# Test with wall time estimation
python hpc/run_potts_sweep.py --config hpc/configs/0_hyperparam_table_test.yaml --estimate_wall_time

# Run actual sweep
python hpc/run_potts_sweep.py --config hpc/configs/0_hyperparam_table_test.yaml
```

### Parameter Precedence

1. **CSV hyperparameters** (highest priority for model-specific params)
   - gamma_factor, beta_factor, poly_order, alpha_rate, r_target, alpha, B_num_vertices, zeta
   - Schedule expressions (beta_schedule, gamma_schedule)

2. **YAML config parameters** (override global settings)
   - T, dt, num_runs, num_states, noise_factor

3. **Default values** (fallback)

### Future Enhancements

- Add `--dry-run` flag to preview which hyperparams would be loaded
- Support merging multiple CSV files
- Add validation to warn if CSV graphs don't match config graphs