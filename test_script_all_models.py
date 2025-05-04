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
T = 1000         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 300000

num_states = 3

noise_factor = 1e-4
seed = 2

#%% Load a coupling graph

file_path = "graphs/band/band250_3_antiferro.col" 
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
#%% Test all model types
poly_order = 3 # order of polynomial model (Polynomial model)
alpha = -1.0 # alpha value for sigmoid model (Sigmoid model)
alpha_rate = 1e-2 # rate of change of alpha (NEC model)
gamma = 1 # gamma value for sigmoid model (NEC model)
r_target = 2 # target radius for NEC model
initial_alpha_arr = 1 * np.ones(num_spins) # initial alpha values for NEC model
beta_schedule  = np.linspace(0, 1.0, num_steps) # beta schedule (Polynomial and Sigmoid models)
gamma_schedule = beta_schedule * 2 # gamma schedule (Polynomial, Sigmoid, and Fixed-Amplitude models)

print(f"Best‐only mode = {best_only}\n")

print("Running polynomial model...")
start = time.time()
res_poly = potts_sim.run_polynomial(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed,
    poly_order,
    list(beta_schedule), list(gamma_schedule),
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True,
    return_cut_value=True,
    return_best_only=best_only
)
duration = time.time() - start
# extract best values
energies = np.array(res_poly["energy"])
cuts     = np.array(res_poly["cut_value"])
best_e = energies[0] if best_only else energies.min()
best_c = cuts[0]     if best_only else cuts.max()
print(f"Polynomial best energy = {best_e} (opt {opt_energy}, diff {best_e-opt_energy})")
print(f"Polynomial best cut   = {best_c} (opt {opt_cut},   diff {best_c-opt_cut})")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

# repeat for NEC
print("Running NEC model...")
start = time.time()
res_nec = potts_sim.run_nec(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed,
    alpha_rate, gamma, r_target,
    list(initial_alpha_arr),
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True,
    return_cut_value=True,
    return_best_only=best_only
)
duration = time.time() - start
energies = np.array(res_nec["energy"])
cuts     = np.array(res_nec["cut_value"])
best_e   = energies[0] if best_only else energies.min()
best_c   = cuts[0]     if best_only else cuts.max()
print(f"NEC best energy        = {best_e} (opt {opt_energy}, diff {best_e-opt_energy})")
print(f"NEC best cut           = {best_c} (opt {opt_cut},   diff {best_c-opt_cut})")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

# repeat for Sigmoid
print("Running sigmoid model...")
start = time.time()
res_sig = potts_sim.run_sigmoid(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed, alpha,
    list(beta_schedule), list(gamma_schedule),
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True,
    return_cut_value=True,
    return_best_only=best_only
)
duration = time.time() - start
energies = np.array(res_sig["energy"])
cuts     = np.array(res_sig["cut_value"])
best_e   = energies[0] if best_only else energies.min()
best_c   = cuts[0]     if best_only else cuts.max()
print(f"Sigmoid best energy    = {best_e} (opt {opt_energy}, diff {best_e-opt_energy})")
print(f"Sigmoid best cut       = {best_c} (opt {opt_cut},   diff {best_c-opt_cut})")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

# repeat for Fixed-Amplitude
print("Running fixed_amplitude model...")
start = time.time()
res_fix = potts_sim.run_fixed_amplitude(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed,
    list(gamma_schedule),
    return_continuous_states=True,
    return_discrete_states=True,
    return_energy=True,
    return_cut_value=True,
    return_best_only=best_only
)
duration = time.time() - start
energies = np.array(res_fix["energy"])
cuts     = np.array(res_fix["cut_value"])
best_e   = energies[0] if best_only else energies.min()
best_c   = cuts[0]     if best_only else cuts.max()
print(f"Fixed-Amplitude best energy = {best_e} (opt {opt_energy}, diff {best_e-opt_energy})")
print(f"Fixed-Amplitude best cut     = {best_c} (opt {opt_cut},   diff {best_c-opt_cut})")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

#%% Plot all model results (skip when best_only)
if not best_only:
    def plot_model(res, name):
        cont   = res["continuous_states"]
        disc   = res["discrete_states"]
        energy = res["energy"]
        cut    = res["cut_value"]
        t_full = np.arange(num_steps) * dt

        # downsample for plotting
        if cont.shape[0] > 1000:
            idxs = np.linspace(0, cont.shape[0] - 1, 1000, dtype=int)
        else:
            idxs = np.arange(cont.shape[0])
        t   = t_full[idxs]
        cont= cont[idxs]
        disc= disc[idxs]
        energy = energy[idxs]
        cut    = cut[idxs]

        # continuous: amplitude & phase
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
        amp   = np.abs(cont)
        phase = np.angle(cont)
        for i in range(num_spins):
            ax1.plot(t, amp[:,i],   lw=0.8)
            ax2.plot(t, phase[:,i], lw=0.8)
        ax1.set_ylabel("Amplitude")
        ax2.set_ylabel("Phase")
        ax2.set_xlabel("Time")
        fig.suptitle(f"{name} Continuous States")
        plt.tight_layout()

        # discrete states
        fig, ax = plt.subplots(figsize=(10,4))
        for i in range(num_spins):
            ax.plot(t, disc[:,i], lw=0.8)
        ax.set_ylabel("Discrete State")
        ax.set_xlabel("Time")
        fig.suptitle(f"{name} Discrete States")
        plt.tight_layout()

        # energy
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(t, energy)
        if opt_energy is not None:
            ax.axhline(opt_energy, color='red', linestyle='--', label='Optimum Energy')
            ax.legend()
        ax.set_ylabel("Energy")
        ax.set_xlabel("Time")
        fig.suptitle(f"{name} Energy")
        plt.tight_layout()

        # cut value
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(t, cut)
        if opt_cut is not None:
            ax.axhline(opt_cut, color='red', linestyle='--', label='Optimum Cut Value')
            ax.legend()
        ax.set_ylabel("Cut Value")
        ax.set_xlabel("Time")
        fig.suptitle(f"{name} Cut Value")
        plt.tight_layout()

    # invoke plots
    plot_model(res_poly, "Polynomial")
    plot_model(res_nec_test, "NEC")
    plot_model(res_sig, "Sigmoid")
    plot_model(res_fix, "Fixed-Amplitude")
    plt.show()

# %%
