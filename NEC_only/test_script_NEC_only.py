#%% Import 
import numpy as np
import matplotlib.pyplot as plt
import time

to_build = False
if to_build:
    import subprocess
    import sys
    import os

    # Delete the build folder if it exists
    build_folder = os.path.join(os.path.dirname(__file__), 'build')
    if os.path.exists(build_folder):
        import shutil
        shutil.rmtree(build_folder)

    setup_path = os.path.join(os.path.dirname(__file__), 'setup.py')
    # Build extension in place so potts_sim.*.so or .pyd is created in the same folder
    subprocess.check_call([sys.executable, setup_path, 'build_ext', '--inplace'])

import potts_sim # Import the custom module

#%% Function to parse graph files

def parse_graph(file_path,zero_based=False):
    """
    Parses a graph file in DIMACS format to extract the number of spins, number of connections,
    and the connections between spins (by pairs, no repeats).

    Parameters:
        file_path (str): The path to the input file.
        
    Returns:
        num_spins (int): Number of spins extracted from the 'p' line.
        num_connections (int): Number of connections extracted from the 'p' line.
        connections (np.ndarray): A 2xN array with sources and targets.
    """
    sources = []
    targets = []
    coupling_strengths = []  # Collected for potential future use
    
    num_spins = None
    num_connections = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Extract the parameters from the line starting with 'p'
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 3:
                    num_spins = int(parts[1])
                    num_connections = int(parts[2])
            # Extract edge information from lines starting with 'e'
            elif line.startswith("e"):
                parts = line.split()
                if len(parts) >= 4:
                    sources.append(int(parts[1]) if zero_based else (int(parts[1]) - 1))
                    targets.append(int(parts[2]) if zero_based else (int(parts[2]) - 1))
                    coupling_strengths.append(int(parts[3]))
    
    # Combine sources and targets into a 2 x N numpy array
    connections = np.array([sources, targets])
    
    return num_spins, num_connections, connections
#%% Set up the simulation parameters

# Example parameters
T = 20         # total simulation time
dt = 1e-3       # time step

num_states = 4

alpha_rate = 1e-2
gamma = 0.5
r_target = 2
noise_factor = 1e-4
seed = 2

#%% Load a coupling graph

file_path = "DSJC250.9.col" 
num_spins, num_connections, connections = parse_graph(file_path)

# Define the initial alpha values for each spin
initial_alpha_arr = 1 * np.ones(num_spins)

#%% Run the simulation

start_time = time.time() # Start timer
result = potts_sim.run(
    T, dt, num_spins, num_states,
    alpha_rate, gamma, r_target,
    connections,
    initial_alpha_arr,
    noise_factor,
    seed,
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True
)

num_steps = np.floor(T / dt)
end_time = time.time() # End timer
duration = end_time - start_time
time_per_step = (duration * 1e6) / num_steps

print(f"Simulation completed in {duration:.2f} seconds.")
print(f"Time per step: {time_per_step:.2f} μs")

#%% Extract histories
cont_hist = result["continuous_states"]    # complex array (num_steps, num_spins)
disc_hist = result["discrete_states"]      # int   array (num_steps, num_spins)
energy_hist = result["energy"]             # float array (num_steps,)

time_array = np.arange(num_steps) * dt

#%% Plot continuous states (amplitude & phase)
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
amp   = np.abs(cont_hist)
phase = np.angle(cont_hist)
for i in range(num_spins):
    ax1.plot(time_array, amp[:, i],   lw=0.8)
    ax2.plot(time_array, phase[:, i], lw=0.8)
ax1.set_ylabel("Amplitude")
ax1.set_title("Spin Amplitudes vs Time")
ax2.set_ylabel("Phase")
ax2.set_xlabel("Time")
ax2.set_title("Spin Phases vs Time")
plt.tight_layout()

#%% Plot discrete states
fig2, ax3 = plt.subplots(figsize=(10, 4))
for i in range(num_spins):
    ax3.plot(time_array, disc_hist[:, i], lw=0.8, label=f"Spin {i}" if num_spins <= 10 else None)
ax3.set_ylabel("Discrete State")
ax3.set_xlabel("Time")
ax3.set_title("Discrete States vs Time")
if num_spins <= 10:
    ax3.legend(loc="upper right", ncol=2)
plt.tight_layout()

#%% Plot system energy
fig3, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(time_array, energy_hist)
ax4.set_xlabel("Time")
ax4.set_ylabel("Energy")
ax4.set_title("System Energy vs Time")
plt.tight_layout()

plt.show()

# %%
