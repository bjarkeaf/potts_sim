#!/usr/bin/env bash
set -e

# source virtualenv if needed
# source /path/to/venv/bin/activate

# direct Python invocation (no MPI)
python3 run_potts_sweep.py --config configs/local_test.yaml
