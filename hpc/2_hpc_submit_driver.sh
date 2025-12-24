# submit_driver.sh
#BSUB -J m3_anneal           # job name
#BSUB -q hpc                    # queue name
#BSUB -n 120                     # total cores (max 128 for hpc, max 360 for fotonano)
#BSUB -R "span[ptile=12]"       # max cores/node (max 20 for hpc)
#BSUB -R "rusage[mem=4GB]"	# memory per core (2GB -> fast alloc)
#BSUB -W 72:00                  # max walltime (max 72 hours for hpc, max 120 hours for fotonano)
#BSUB -u s194084@dtu.dk         # email address
#BSUB -B                        # send email at start
#BSUB -N                        # send email at end
#BSUB -o logs/%J.out     # output file
#BSUB -e logs/%J.err     # error file

module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6
source potts-env/bin/activate

mpirun python3 run_potts_sweep.py --config configs/251220_G1-5_max-3-cut_convergence_annealing.yaml
