import numpy as np
from mpi4py import MPI
from itertools import chain
import potts_sim
from graph_parser import parse_graph

#%% Set up the simulation parameters

T = 20         # total simulation time
dt = 1e-3       # time step

alpha_rate = 1e-2
gamma = 1
r_target = 2
noise_factor = 1e-4

#%% Load a coupling graph

file_path = "DSJC250.9.col" 
num_spins, num_edges, edges, opt_cut, opt_energy = parse_graph(file_path)

# Define the initial alpha values for each spin
initial_alpha_arr = 1 * np.ones(num_spins)

# Define Potts q‐state
num_states = 3

#%% HPC setup and running

if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # total runs & assign each rank a strided slice of seeds
    num_runs = 100
    my_seeds = list(range(num_runs))[rank::size]

    local_data = []
    for seed in my_seeds:
        res = potts_sim.run_nec(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, seed,
            alpha_rate, gamma, r_target,
            initial_alpha_arr.tolist(),
            return_continuous_states=False,
            return_discrete_states=True
        )
        # take the last discrete state
        last_state = res["discrete_states"][-1]
        local_data.append((seed, last_state))

    print(f"Rank {rank} finished processing.")
    
    # Gather all data at rank 0
    all_data = comm.gather(local_data, root=0)
    if rank == 0:
        # flatten and sort by seed
        flat = list(chain.from_iterable(all_data))
        flat.sort(key=lambda x: x[0])

        # build a single array of shape (num_runs, num_spins)
        final_states = np.vstack([state for (_, state) in flat])

        # save once
        np.save("results/final_states.npy", final_states)
        print("Simulation results saved.")

