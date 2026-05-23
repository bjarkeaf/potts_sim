#!/bin/bash
# submit_template.sh -- LSF job submission template for potts_sim
#
# Usage:
#   1. Copy this file and edit the parameters below for your run.
#   2. Update the --config path to point to your YAML configuration.
#   3. Submit with: bsub < your_submit_script.sh
#
# Queue options (DTU gbar):
#   hpc       -- up to 128 cores/node, max 72-hour walltime
#   fotonano  -- up to 360 cores/node, max 120-hour walltime
#   amd       -- AMD nodes

#BSUB -J potts_sweep            # job name
#BSUB -q hpc                    # queue name
#BSUB -n 60                     # total MPI ranks (tune to your sweep size)
#BSUB -R "span[ptile=60]"       # cores per node
#BSUB -R "rusage[mem=4GB]"      # memory per core
#BSUB -W 24:00                  # max walltime hh:mm
#BSUB -u YOUR_EMAIL@example.com # notification email
#BSUB -B                        # email at job start
#BSUB -N                        # email at job end
#BSUB -o logs/driver.%J.out     # stdout log
#BSUB -e logs/driver.%J.err     # stderr log

module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6

source potts-env/bin/activate

mpirun python3 run_potts_sweep.py --config configs/YOUR_CONFIG.yaml
