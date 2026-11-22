#!/bin/bash
# submit_template_slurm.sh -- Slurm job submission template for potts_sim
#
# Usage, from the repo root:
#   1. Edit the module lines below to match the modules used when building (see README, Installation).
#   2. Submit with the config as argument, overriding the walltime if needed:
#        sbatch submit_template_slurm.sh configs/YOUR_CONFIG.yaml
#        sbatch --time=01:00:00 submit_template_slurm.sh configs/YOUR_CONFIG.yaml
#
# A job killed at its walltime can be resumed by submitting the same command again.

#SBATCH --job-name=potts_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96    # MPI ranks: set to the number of cores per node
#SBATCH --time=04:00:00         # max walltime hh:mm:ss
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

CONFIG=${1:?usage: sbatch submit_template_slurm.sh configs/YOUR_CONFIG.yaml}
VENV=${VENV:-potts-env}         # virtual environment created during installation

# module load ...               # same Python, GCC and MPI modules as used for the build

source "$VENV/bin/activate"

# mpirun picks up the number of ranks from the Slurm allocation. Use srun or the
# cluster's recommended launcher instead (e.g. mympirun at HPC-UGent) if preferred.
mpirun python run_potts_sweep.py --config "$CONFIG"
