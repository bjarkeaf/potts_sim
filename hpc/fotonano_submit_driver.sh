# submit_driver.sh
#BSUB -J g05-dt           # job name
#BSUB -q fotonano                    # queue name
#BSUB -n 100                     # total cores (max 128 for hpc, max 360 for fotonano)
#BSUB -R "span[ptile=20]"       # max cores/node (max 20 for hpc)
#BSUB -R "rusage[mem=6GB]"	# memory per core (2GB -> fast alloc)
#BSUB -W 120:00                  # max walltime (max 72 hours for hpc, max 120 hours for fotonano)
#BSUB -u s194084@dtu.dk         # email address
#BSUB -B                        # send email at start
#BSUB -N                        # send email at end
#BSUB -o logs/driver.%J.out     # output file
#BSUB -e logs/driver.%J.err     # error file

module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6
source potts-env/bin/activate

mpirun python3 run_potts_sweep.py --config configs/251226_g05_convergence_time_step.yaml
