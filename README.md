# Potts Machine Simulator

Research code for the paper **"Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut"** (`readme.pdf`).

The repository implements and benchmarks five analog Potts machine (PM) models -- NEC, q-PDC, q-SHIL, Polynomial PM, Sigmoid PM -- and one reference Ising machine (Reference IM) for solving Max-3-Cut and Max-4-Cut combinatorial optimization problems on the G-set benchmark graphs. Simulations use an Euler-Maruyama ODE solver implemented in C++ (via pybind11) and are parallelised with MPI for large-scale parameter sweeps on HPC clusters.

## Paper

> Bjarke Almer Frederiksen, "Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut", 2026. See `readme.pdf`.

## Requirements

- Python 3.10+
- A C++ compiler (GCC or Clang) for building the simulation extension
- An MPI implementation (e.g. OpenMPI) for multi-process sweeps

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Installation

Build the C++ simulation extension:

```bash
python build_potts_sim.py
```

This produces `potts_sim.cpython-*.so` in the project root, which is imported by the Python scripts.

## Quickstart

### Local test (single process)

```bash
python hpc/run_potts_sweep.py --config hpc/configs/0_local_test.yaml
```

### Estimate wall time before a large sweep

```bash
python hpc/run_potts_sweep.py --config hpc/configs/YOUR_CONFIG.yaml --estimate_wall_time 72
```

### Visualise parameter schedules

```bash
python hpc/run_potts_sweep.py --config hpc/configs/YOUR_CONFIG.yaml --plot_schedules
```

### HPC cluster (LSF)

```bash
cd hpc/
# Edit submit_template.sh: set your email, core count, walltime, and config path
bsub < submit_template.sh

# Monitor job
bstat
tail -f logs/driver.<JOBID>.out
```

Results are written as Parquet files to `hpc/results/` and can be merged with `merge_parquet.py`.

## Repository Structure

```
potts_sim/
├── potts_sim.cpp          # C++ simulation engine (Euler-Maruyama, all models)
├── build_potts_sim.py     # Builds the C++ extension via pybind11
├── potts_utils.py         # Graph parsing (DIMACS) and eigenvalue utilities
├── cim_sim.py             # Coherent Ising Machine simulator wrapper
├── readme.pdf             # Associated paper
├── graphs/                # Input graphs in DIMACS .col format (G-set + g05)
├── figures/               # Publication figures and generation scripts
└── hpc/
    ├── run_potts_sweep.py     # Main entry point; MPI sweep runner
    ├── configs/               # YAML experiment configurations
    ├── best_hyperparams/      # Saved optimal hyperparameters per model/graph
    ├── submit_template.sh     # LSF job submission template
    ├── merge_parquet.py       # Merges per-rank result files
    ├── plot_benchmark.py      # Benchmark result visualisation
    ├── plot_convergence.py    # Convergence analysis plots
    ├── save_best_hyperparams.py
    ├── results/               # Output Parquet files (gitignored)
    └── logs/                  # Job stdout/stderr logs (gitignored)
```

## Configuration

Experiments are defined in YAML files under `hpc/configs/`. The naming convention is `YYMMDD_<graph_set>_<model>.yaml`.

Key fields:

| Field | Description |
|---|---|
| `num_runs` | Independent simulation runs per parameter combination |
| `graph_path` | List of DIMACS graph files |
| `num_states` | Number of Potts states (3 for Max-3-Cut, 4 for Max-4-Cut) |
| `T` / `dt` | Simulation duration and time step |
| `noise_factor` | Stochastic noise amplitude |
| `models` | Per-model parameter sweeps (supports ranges like `"0:0.5:2"` and schedule expressions like `"lin(1/mu_max, 1.0)"`) |

See `hpc/configs/0_local_test.yaml` for a minimal working example.

## Citation

If you use this code, please cite:

```bibtex
@article{frederiksen2026potts,
  title   = {Comparative Study of Potts Machine Dynamics and Performance for Max-k-Cut},
  author  = {Frederiksen, Bjarke Almer},
  year    = {2026},
}
```

## License

MIT -- see `LICENSE`.
