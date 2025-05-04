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

import potts_sim # Import the custom module

#%% Set up the simulation parameters

# Example parameters
T = 100         # total simulation time
dt = 0.01       # time step
num_spins = 2500 # small system

alpha_rate = 0.5
gamma = 5
r_target = 5
num_states = 3
noise_factor = 1e-4

# Define the initial alpha values for each spin
initial_alpha_list = - 4 * np.ones(num_spins)

#%% Generate a coupling graph (2D lattice with periodic boundary conditions)
L = int(np.sqrt(num_spins))
connection_list = []

for i in range(num_spins):
    # Convert vector index to 2D coordinates in row-major order
    row = i // L
    col = i % L
    
    # Determine neighbor coordinates with periodic boundary conditions
    row_up = (row - 1) % L
    row_down = (row + 1) % L
    col_left = (col - 1) % L
    col_right = (col + 1) % L
    
    # Map 2D coordinates back to vector indices (row-major ordering)
    up = row_up * L + col
    down = row_down * L + col
    left = row * L + col_left
    right = row * L + col_right
    
    # Add connections for this spin
    neighbors = [up, down, left, right]
    connection_list.append(neighbors)

#%% Run the simulation

start_time = time.time() # Start timer
result = potts_sim.simulate(
    T, dt, num_spins,
    alpha_rate, gamma, r_target,
    connection_list,
    initial_alpha_list,
    num_states,
    noise_factor
)
end_time = time.time() # End timer
duration = end_time - start_time
num_steps = int(T / dt)

print(f"Elapsed time: {duration:.2f} s")
print(f'Time per step: {duration/num_steps*1e6:.2f} μs')


#%% Plot the results
time_array = np.arange(num_steps) * dt

# Separate amplitude and phase
amplitude = np.abs(result)
phase = np.angle(result)

# Create figure for amplitudes and phases
fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot amplitude for each spin
for spin in range(num_spins):
    ax_amp.plot(time_array, amplitude[:, spin], lw=0.8)
ax_amp.set_ylabel("Amplitude")
ax_amp.set_title("Spin Amplitudes vs Time")

# Plot phase for each spin
for spin in range(num_spins):
    ax_phase.plot(time_array, phase[:, spin], lw=0.8)
ax_phase.set_xlabel("Time")
ax_phase.set_ylabel("Phase")
ax_phase.set_title("Spin Phases vs Time")

plt.tight_layout()
plt.show()

# %%
