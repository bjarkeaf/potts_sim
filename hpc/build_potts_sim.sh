module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6
source potts-env/bin/activate

python3 build_potts_sim.py build_ext --inplace # Build

rm -r -f build # Remove build folder

mv potts_sim.*.so potts-env/lib/python3.13/site-packages/ # Move to python environment
