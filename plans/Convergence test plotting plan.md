# Plan for Convergence Test Plotting Script

This document describes the requirements for a script in the 'hpc' folder called 'plot_convergence.py'.

## Purpose

The script generates plots showing how a figure of merit (FOM) converges as a function of a swept parameter (simulation time T, time step dt, and later also other parameters). This helps determine optimal simulation parameters and verify that results have converged to stable values.

**Key behavior:**

- Generates **one plot file per model** found in the data
- Each plot contains subplots based on `--graph_grouping` (per graph, all graphs combined, or by graph size)
- X-axis: swept parameter (T or dt)
- Y-axis: figure of merit (mean_gap or success_rate)

**Reference:** See `hpc/plot_benchmark.py` (especially `plot_rel_gap_distributions_by_graph`) for example plotting code.

## Arguments

### Required Arguments

- `--data <path>` **(required)**
  - Path to Parquet file containing convergence sweep results
  - File format: Parquet with columns including `graph_path`, `model_type`, `T`, `dt`, `best_cut_val`, etc.
  - Example: `hpc/results/250630_convergence_sweep.parquet`

- `--conv_type <type>` **(required)**
  - Specifies which parameter is swept (x-axis)
  - Options:
    - `simulation_time`: Total simulation time T
    - `time_step`: Time step size dt
  - Future options could include annealing speed parameters.

### Optional Arguments

- `--fom <metric>` (default: `mean_gap`)
  - Figure of merit to plot on y-axis
  - Options:
    - `mean_gap`: Mean relative optimality gap across runs
      - Formula: `mean((optimal_cut - best_cut_val) / optimal_cut * 100)`
      - Lower is better (0% = optimal, positive values indicate suboptimal solutions)
    - `success_rate`: Percentage of runs that found the ground state
      - Formula: `sum(best_cut_val == optimal_cut) / num_runs * 100`
      - Range: [0, 100]%

- `--graph_grouping <mode>` (default: `per_graph`)
  - Controls subplot organization
  - Options:
    - `per_graph`: One subplot per unique graph file
      - Use when analyzing specific graph instances
    - `all_graphs`: Single subplot averaging over all graphs
      - Use for overall convergence behavior
    - `by_graph_size`: Group graphs by number of vertices
      - Use to compare convergence across problem sizes

- `--output_dir <path>` (default: `hpc/plots/{data name, without results_ prefix}`)
  - Directory to save output plots
  - Created automatically if it doesn't exist

## Plot Types by FOM and Grouping

The visualization type depends on the combination of `--fom` and `--graph_grouping`:

### FOM: `mean_gap`

**All grouping modes:**

- Use **boxplots** showing the distribution of optimality gaps across simulation runs
- X-axis: swept parameter values (e.g., T or dt)
- Y-axis: relative optimality gap (%)
- Each x-position shows a box representing the distribution over `num_runs` independent runs
- Grouping affects what data goes into each box:
  - `per_graph`: Box contains gaps from runs on that specific graph
  - `all_graphs`: Box contains gaps from runs on all graphs combined
  - `by_graph_size`: Box contains gaps from runs on all graphs of that size
- Add a horizontal reference line at y=0 (optimal solution)
- Consider adding mean as a marker (e.g., diamond) overlaid on boxes

### FOM: `success_rate`

**Grouping: `per_graph`**
- Use **scatter/line plots** (one trace per graph)
- X-axis: swept parameter values
- Y-axis: success rate (%)
- Each point represents: `(swept_param_value, success_rate_for_that_value)`
- Connect points with lines for easier trend visualization
- Add markers at data points

**Grouping: `all_graphs` or `by_graph_size`**
- Use **boxplots** showing the distribution of success rates across graphs
- X-axis: swept parameter values
- Y-axis: success rate (%)
- Each box shows the distribution of success rates over multiple graphs
- This reveals how consistently different graphs converge at each parameter value

## Data Processing

### Loading and Parsing
1. Load Parquet file using `pd.read_parquet(args.data)`
2. Extract optimal cut values from graph metadata (parse DIMACS comment lines)
3. Group data by:
   - Model type (e.g., POLYNOMIAL, SIGMOID, NEC, CIM)
   - Graph path or graph size (depending on grouping mode)
   - Swept parameter value (T or dt)

### Metric Calculation
- **Optimality gap:** `(optimal_cut - best_cut_val) / optimal_cut * 100`
  - Always positive: 0% means optimal solution found, higher values indicate worse performance
  - Handle graphs without known optimal: skip or warn user
- **Success rate:** `(best_cut_val == optimal_cut).sum() / len(best_cut_val) * 100`
  - Use exact equality check (cut values are integers)

### Graph Size Extraction
For `by_graph_size` grouping:
- Parse number of vertices from DIMACS `p` line: `p <n_vertices> <n_edges> <weight>`
- Group graphs with identical vertex counts

## Plot Aesthetics

### Figure Layout
- **Subplot grid:**
  - Determine number of subplots based on grouping mode
  - Use `plt.subplots()` with appropriate `nrows` and `ncols`
  - Suggested layout: max 3 columns, adjust rows as needed
  - Figure size: scale with number of subplots (e.g., `figsize=(5*ncols, 4*nrows)`)

- **Spacing:**
  - `plt.tight_layout()` or `plt.subplots_adjust()` to prevent overlap
  - Add space for suptitle: `top=0.95`

### Styling
- **Colors:**
  - Use consistent color scheme across plots
  - For multiple graphs in one subplot: use distinct colors from `plt.cm.tab10` or similar
  - For boxplots: can use default matplotlib styling or custom palette

- **Labels:**
  - X-axis label based on `conv_type`:
    - `simulation_time`: "Simulation time, T"
    - `time_step`: "Time step, dt"
  - Y-axis label based on `fom`:
    - `mean_gap`: "Relative optimality gap (%)"
    - `success_rate`: "Success rate (%)"
  - Subplot titles:
    - `per_graph`: graph filename (e.g., "g05_0.col")
    - `by_graph_size`: "Graph size: N vertices"
    - `all_graphs`: "All graphs combined"

- **Legend:**
  - Include legend when multiple traces in one subplot
  - Position: `best` or `upper right` to avoid obscuring data
  - For `per_graph` with single graph: legend may be omitted

- **Grid:**
  - Add light gridlines: `ax.grid(True, alpha=0.3, linestyle='--')`

### Figure Title
- Main title (suptitle) should include:
  - Model type
  - FOM being analyzed
  - Convergence parameter
  - Example: "NEC Model: Success Rate vs Simulation Time T"

## Output Files

### Naming Convention
Format: `convergence_<conv_type>_<fom>_<model>_<grouping>.png`

Examples:
- `convergence_simulation_time_mean_gap_POLYNOMIAL_per_graph.png`
- `convergence_time_step_success_rate_CIM_by_graph_size.png`
- `convergence_simulation_time_mean_gap_NEC_all_graphs.png`

### File Format
- **Primary format:** PNG with 300 DPI for publication quality
- **Optional:** Also save as PDF for vector graphics
- Save to `args.output_dir`, creating directory if needed

### Console Output
Print status messages:
```
Loading data from: hpc/results/250630_convergence_sweep.parquet
Found models: POLYNOMIAL, SIGMOID, NEC
Processing POLYNOMIAL model...
  - 5 graphs found
  - Swept parameter: T in [10, 20, 50, 100, 200, 500]
  - Generated plot: hpc/plots/convergence/convergence_simulation_time_mean_gap_POLYNOMIAL_per_graph.png
Processing SIGMOID model...
  ...
Done. Generated 3 plots.
```

## Error Handling and Edge Cases

### Data Validation
- **Empty data:** Error if no data loaded or no data matches filters
- **Missing columns:** Check for required columns (`graph_path`, `best_cut_val`, `T`, `dt`, model-specific columns)
- **No optimal values:** Warn and skip graphs without known optimal cut values for gap calculations

### Convergence Parameter Validation
- Verify that `conv_type` parameter is actually swept in the data
- Error if all rows have the same value for the swept parameter
- Warn if very few (< 3) unique values present

### Numerical Edge Cases
- **Division by zero:** Can occur if `optimal_cut == 0` (unlikely but possible)
  - Handle gracefully or skip such graphs
- **All runs succeed:** Success rate = 100% (valid, no special handling)
- **All runs fail:** Success rate = 0% (valid, no special handling)

### Plot Edge Cases
- **Single data point:** Boxplot with one point still valid, but warn user
- **Very many subplots:** Warn if > 20 subplots (figure may be too large)
- **No data for subplot:** Skip subplot or show empty with note

## Implementation Notes

### Dependencies
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
```

### Code Structure
Suggested functions:
1. `parse_args()` → Parse command-line arguments
2. `load_data(parquet_path)` → Load and validate Parquet file
3. `extract_optimal_values(graph_paths)` → Parse DIMACS files for optimal cuts
4. `calculate_metrics(df, fom, optimal_dict)` → Compute gaps or success rates
5. `get_graph_sizes(graph_paths)` → Extract number of vertices
6. `create_convergence_plot(data, conv_type, fom, grouping, model, output_dir)` → Main plotting logic
7. `main()` → Orchestrate workflow

### Testing Checklist
- [ ] Test with `mean_gap` and `success_rate` FOMs
- [ ] Test all three grouping modes
- [ ] Test with both `simulation_time` and `time_step` convergence types
- [ ] Test with multiple model types in one dataset
- [ ] Test with missing optimal values
- [ ] Verify output file naming is correct
- [ ] Check plot aesthetics (labels, titles, legends)

### Future Enhancements

- Add `--models` filter to plot only specific models
- Add `--graphs` filter to plot only specific graphs
- Support for additional FOMs (e.g., median_gap, std_gap)
- Interactive plots using plotly
- Statistical annotations (e.g., convergence threshold lines)
- Comparison mode: overlay multiple convergence types on same plot