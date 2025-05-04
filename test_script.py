#%% Import 
import numpy as np
import matplotlib.pyplot as plt
import time

to_build = True
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

import potts_sim  # Import the custom module
from graph_parser import parse_graph

#%% Set up the simulation parameters

# Example parameters
T = 100         # total simulation time
dt = 1e-3       # time step
num_steps = np.floor(T / dt)  # number of time steps

num_states = 3

alpha_rate = 1e-2
gamma = 1
r_target = 2
noise_factor = 1e-4
seed = 2

#%% Load a coupling graph

file_path = "DSJC250.9.col" 
num_spins, num_edges, edges = parse_graph(file_path)

# Extract optimums from comments
opt_cut = None
opt_energy = None
with open(file_path, 'r') as f:
    for line in f:
        if line.startswith("c Optimum cut value (max3cut):"):
            opt_cut = int(line.split(":")[1])
        elif line.startswith("c Optimum energy (max3cut):"):
            opt_energy = int(line.split(":")[1])

# Define the initial alpha values for each spin
initial_alpha_arr = 1 * np.ones(num_spins)

#%% Run the simulation

start_time = time.time()  # Start timer
result = potts_sim.run_nec(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed,
    alpha_rate, gamma, r_target,
    initial_alpha_arr,
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True,
    return_cut_value=True
)
end_time = time.time()  # End timer
duration = end_time - start_time
num_steps = int(T / dt)

print(f"Simulation completed in {duration:.2f} seconds.")
print(f"Time per step: {duration * 1e6 / num_steps:.2f} μs")

#%% Extract histories
cont_hist = result["continuous_states"]    # complex array (num_steps, num_spins)
disc_hist = result["discrete_states"]      # int   array (num_steps, num_spins)
energy_hist = result["energy"]             # float array (num_steps,)
cut_hist = result["cut_value"]             # new

time_array = np.arange(num_steps) * dt

#%% Plotting 

# Plot continuous states (amplitude & phase)
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

# Plot discrete states
fig2, ax3 = plt.subplots(figsize=(10, 4))
for i in range(num_spins):
    ax3.plot(time_array, disc_hist[:, i], lw=0.8, label=f"Spin {i}" if num_spins <= 10 else None)
ax3.set_ylabel("Discrete State")
ax3.set_xlabel("Time")
ax3.set_title("Discrete States vs Time")
if num_spins <= 10:
    ax3.legend(loc="upper right", ncol=2)
plt.tight_layout()

# Plot system energy
fig3, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(time_array, energy_hist)
if opt_energy is not None:
    ax4.axhline(opt_energy, color='red', linestyle='--', label='Optimum Energy')
    ax4.legend()
ax4.set_xlabel("Time")
ax4.set_ylabel("Energy")
ax4.set_title("System Energy vs Time")
plt.tight_layout()

# Plot cut value separately
fig4, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(time_array, cut_hist)
if opt_cut is not None:
    ax5.axhline(opt_cut, color='red', linestyle='--', label='Optimum Cut Value')
    ax5.legend()
ax5.set_xlabel("Time")
ax5.set_ylabel("Cut Value")
ax5.set_title("Cut Value vs Time")
plt.tight_layout()

plt.show()
