# Potts Machine Simulator

Research code for the paper **"Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut"**.

The repository implements and benchmarks five analog Potts machine (PM) models (NEC, q-PDC, q-SHIL, Polynomial PM, Sigmoid PM) and one reference Ising machine (Reference IM) for solving Max-3-Cut and Max-4-Cut combinatorial optimization problems on the G-set benchmark graphs. Simulations use an Euler-Maruyama ODE solver implemented in C++ (via pybind11) and are parallelised with MPI for large-scale parameter sweeps on HPC clusters.

## Paper

> Bjarke Almer Frederiksen, Robbe De Prins, Peter Bienstman, "Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut", arXiv:2605.06425 (2026). https://doi.org/10.48550/arXiv.2605.06425

## System Requirements

**Operating system:** Linux or macOS. Tested on Linux (Arch, kernel 7.x) with Python 3.14 and GCC 16, and on AlmaLinux 9.8 (DTU HPC cluster). macOS should work with Clang and Homebrew OpenMPI, but has not been formally tested.

**Software dependencies:**

| Dependency | Version tested | Notes |
|---|---|---|
| Python | 3.14 | 3.10+ should work |
| GCC or Clang | GCC 16 | Required to build the C++ extension |
| OpenMPI | 5.0.6 | Required for multi-process sweeps, optional for single-process runs |
| setuptools | 70+ | Installed via pip, required to build the C++ extension |
| pybind11 | 2.13+ | Installed via pip |
| numpy | 2.x | |
| pandas | 2.x | |
| numba | 0.60+ | Required for the CIM model, installed via pip |
| mpi4py | 4.0 | Optional for local single-process runs |

Full Python dependency list is in `requirements.txt`. No special hardware is required.

**Typical install time:** 2 to 5 minutes (dominated by pip dependency installation).

## Installation

Create and activate a virtual environment:

```bash
python -m venv potts-env
source potts-env/bin/activate
```

Install system-level dependencies (required before pip):

- **Linux (Debian/Ubuntu):** `sudo apt install gcc libopenmpi-dev python3-dev`
- **Linux (Arch):** `sudo pacman -S gcc openmpi`
- **macOS:** `brew install open-mpi`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build the C++ simulation extension:

```bash
python build_potts_sim.py build_ext --inplace
```

This produces `potts_sim.cpython-*.so` in the repo root, which is imported by the Python scripts. The build uses `-march=native`, so the compiled binary is CPU-specific and floating-point results may differ slightly across machines.

Verify the build succeeded:

```bash
python -c "import potts_sim; print('OK')"
```

## Demo

The repository includes benchmark graphs in `graphs/` (DIMACS `.col` format). The quickest demo uses two small g05 graphs (10 and 20 nodes) with the Polynomial PM model and 10 runs per parameter combination.

Run from the repo root:

```bash
python run_potts_sweep.py --config configs/0_local_test.yaml
```

Expected output:

```
Found 2 graph files
Generated 60 tasks
Rank 0: Processing 60 tasks
Rank 0: Completed 1/60 tasks | Runtime: 0:00:00 | ETA: 0:00:00 | ...
...
Saved combined results with 60 rows to results/results_0_local_test.parquet
Finished sweep in 0:00:07
```

The result is written to `results/results_0_local_test.parquet` relative to the repo root. The Parquet file contains one row per simulation run. Key columns include `cut_gap` (achieved cut minus optimal cut: zero means the optimum was reached, negative means the solution fell short) and `energy_gap`. To inspect the output:

```bash
python -c "import pandas as pd; print(pd.read_parquet('results/results_0_local_test.parquet')[['model','graph','cut_gap']].to_string())"
```

Expected run time on a standard desktop computer: under 30 seconds (single core).

## Instructions for Use

### Running on your own data

Graphs must be in DIMACS `.col` format (see "Using Your Own Graphs" below). Create a YAML config pointing to your graph files, set `num_states` (3 for Max-3-Cut, 4 for Max-4-Cut), and run:

```bash
python run_potts_sweep.py --config configs/YOUR_CONFIG.yaml
```

### Estimate wall time before a large sweep

```bash
python run_potts_sweep.py --config configs/YOUR_CONFIG.yaml --estimate_wall_time 72
```

The optional argument is the number of MPI ranks to assume for the estimate. Omit it to use the current MPI size (1 for a local run).

### Visualise parameter schedules

```bash
python run_potts_sweep.py --config configs/YOUR_CONFIG.yaml --plot_schedules
```

### HPC cluster (LSF)

```bash
# Edit submit_template.sh: set your email, core count, walltime, and config path
bsub < submit_template.sh

# Monitor job
bstat
tail -f logs/driver.<JOBID>.out
```

Results are written as Parquet files to `results/` and can be merged with `merge_parquet.py`.

## Repository Structure

```
potts_sim/
├── potts_sim.cpp              # C++ simulation engine (Euler-Maruyama, all models)
├── build_potts_sim.py         # Builds the C++ extension via pybind11
├── potts_utils.py             # Graph parsing (DIMACS) and eigenvalue utilities
├── cim_sim.py                 # Coherent Ising Machine simulator wrapper
├── run_potts_sweep.py         # Main entry point, MPI sweep runner
├── save_best_hyperparams.py   # Extracts best hyperparameters from a result file
├── merge_parquet.py           # Merges result files from separate jobs
├── plot_benchmark.py          # Benchmark result visualisation
├── plot_convergence.py        # Convergence analysis plots
├── dynamics_figure.py         # Generates dynamics figures, outputs to figures/
├── submit_template.sh         # LSF job submission template
├── graphs/                    # Input graphs in DIMACS .col format (G-set + g05)
├── figures/                   # Publication figures (PDF and PNG outputs)
├── configs/                   # YAML experiment configurations
├── best_hyperparams/          # Saved optimal hyperparameters per model/graph
├── rudy_to_dimacs.sh          # Converts Rudy-format graphs to DIMACS .col
├── results/
│   └── paper/                 # Final result Parquet files used in the paper
├── logs/                      # Job stdout/stderr logs
└── plots/                     # Generated plot output
```

## Configuration

Experiments are defined in YAML files under `configs/`. The naming convention is `YYMMDD_<graph_set>_<model>.yaml`.

Key fields:

| Field | Description |
|---|---|
| `num_runs` | Independent simulation runs per parameter combination |
| `graph_path` | List of DIMACS graph files |
| `num_states` | Number of Potts states (3 for Max-3-Cut, 4 for Max-4-Cut) |
| `T` / `dt` | Simulation duration and time step |
| `noise_factor` | Stochastic noise amplitude |
| `models` | Per-model parameter sweeps (supports ranges like `"0:0.5:2"` and schedule expressions like `"lin(1/mu_max, 1.0)"`) |

See `configs/0_local_test.yaml` for a minimal working example.

**Schedule expressions** control how parameters vary over simulation time:

| Expression | Meaning |
|---|---|
| `lin(a, b)` | Linear ramp from `a` to `b` over the simulation |
| `linspan(a, s)` | Linear ramp starting at `a` with total span `s` (i.e. ends at `a + s`) |
| `exp(a, b)` | Exponential ramp from `a` to `b` over the simulation |
| `const(x)` | Constant value `x` |
| `mu_max` | Maximum eigenvalue of the coupling matrix (read from the graph file or computed) |
| `"start:step:end"` | MATLAB-style range, generates a sweep list (e.g. `"0:0.5:2"` → [0.0, 0.5, 1.0, 1.5, 2.0]) |

A linked schedule (`based_on` + `factor`) makes one parameter track another multiplied by a factor, which is useful for keeping gamma proportional to beta.

**Model names:** In YAML configs and result files, q-SHIL appears as `FIXED_AMPLITUDE`.

## Using Your Own Graphs

Graphs must be in DIMACS `.col` format:

```
c Maximum eigenvalue: 3.14       (optional, computed on the fly if absent)
c Optimum cut value (max-3-cut): 42  (optional, needed for gap metrics)
p <vertices> <edges> <weight>
e <src> <dst> <weight>
...
```

Edges are 1-indexed. If the `Maximum eigenvalue` comment is absent, it is computed on the first run and written back into the graph file (requires write permission on the file). If the `Optimum cut value` comment is absent, `cut_gap` and `energy_gap` will be null in the output.

A minimal config to run on your own graph:

```yaml
num_runs: 10
graph_path:
  - "path/to/your_graph.col"
out_dir: "results"
T: 100.0
dt: 1e-3
num_states: 3
noise_factor: 1e-4
models:
  POLYNOMIAL:
    poly_order: [3]
    beta_schedule: ["lin(1/mu_max, 1.0)"]
    gamma_schedule:
      based_on: "beta_schedule"
      factor: "1:0.5:2"
```

Graph paths are relative to the working directory from which you run the script. Run from the repo root and use paths like `graphs/my_graph.col`.

## Reproducing Paper Results

The benchmark results in the paper were produced on the DTU Computing Center HPC cluster (LSF scheduler) using the configurations in `configs/`. The final sweep configs are:

| Figure | Config file(s) |
|---|---|
| G-set Max-3-Cut benchmark | `260123_gset_max-3-cut.yaml` |
| G-set Max-4-Cut benchmark | `260123_gset_max-4-cut.yaml` |
| g05 benchmark | `260123_g05.yaml` |
| Convergence (G-set) | `260128_gset_max-3-cut_convergence_*.yaml`, `260128_gset_max-4-cut_convergence_*.yaml` |
| Convergence (g05) | `260128_g05_convergence_*.yaml` |

All commands run from the repo root.

Before submitting, estimate the wall time:

```bash
python run_potts_sweep.py --config configs/260123_gset_max-3-cut.yaml --estimate_wall_time 72
```

**Benchmark workflow** (repeat for each config in the table above):

```bash
# Option A: LSF cluster
# Edit submit_template.sh with your email, core count, and config path, then submit
bsub < submit_template.sh

# Option B: local multi-core (slow for large configs)
mpirun -n 8 python run_potts_sweep.py --config configs/260123_gset_max-3-cut.yaml

# After completion, save best hyperparameters (writes best_hyperparams/260123_gset_max-3-cut_mean_gap.csv)
python save_best_hyperparams.py results/results_260123_gset_max-3-cut.parquet

# Plot benchmark figures
python plot_benchmark.py --results results/results_260123_gset_max-3-cut.parquet --figure_mode
```

**Convergence workflow** (run after the benchmark sweep, as it uses the saved hyperparameters):

```bash
# Submit convergence sweep (edit submit_template.sh to point to a 260128_* config)
bsub < submit_template.sh

# Plot convergence figures
python plot_convergence.py \
    --data results/results_260128_gset_max-3-cut_convergence_sim_time.parquet \
    --conv_type simulation_time
```

`merge_parquet.py` is only needed when combining results from separate jobs manually. The sweep auto-merges results on completion.

Saved optimal hyperparameters used in the paper are provided in `best_hyperparams/` for reference.

The full benchmark sweeps require a multi-core HPC cluster and take on the order of tens of CPU-hours per config. Single-node reproduction is possible but slow.

## Citation

If you use this code, please cite:

```bibtex
@misc{frederiksen2026potts,
  title         = {Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut},
  author        = {Frederiksen, Bjarke Almer and De Prins, Robbe and Bienstman, Peter},
  year          = {2026},
  eprint        = {2605.06425},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.stat-mech},
  doi           = {10.48550/arXiv.2605.06425},
}
```

## License

MIT. See `LICENSE`.
