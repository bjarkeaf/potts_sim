# submit_driver.sh
#BSUB -J potts-driver           # job name
#BSUB -q hpc                    # queue name
#BSUB -n 40                     # total cores
#BSUB -R "span[ptile=20]"       # max cores/node
#BSUB -R "rusage[mem=4GB]"	    # memory per core
#BSUB -W 72:00                  # max walltime
#BSUB -u s194084@dtu.dk         # email address
#BSUB -B                        # send email at start
#BSUB -N                        # send email at end
#BSUB -o logs/driver.%J.out     # output file
#BSUB -e logs/driver.%J.err     # error file

module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6
source potts-env/bin/activate

mpirun python3 run_potts_sweep.py --config sweep_config.yaml
