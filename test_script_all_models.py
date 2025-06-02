#%% Import 
import numpy as np
import matplotlib.pyplot as plt
import time

#to_build = True
to_build = False
if to_build:
    import subprocess
    import sys
    import os
    import shutil

    # Delete the build folder if it exists
    build_folder = os.path.join(os.path.dirname(__file__), 'build')
    if os.path.exists(build_folder):
        shutil.rmtree(build_folder)

    setup_path = os.path.join(os.path.dirname(__file__), 'build_potts_sim.py')
    # Build extension in place so potts_sim.*.so or .pyd is created in the same folder
    subprocess.check_call([sys.executable, setup_path, 'build_ext', '--inplace'])

    shutil.rmtree(build_folder) # Clean up build folder
    
import potts_sim  # Import the custom module
from potts_utils import parse_graph

#%% Define plotting function
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

#%% Set up the simulation parameters

# Default parameters
T = 20         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 300000

num_states = 3

noise_factor = 1e-4
seed = 2

#%% Load a coupling graph

# parse_graph now returns opt_cut, opt_energy
#file_path = "graphs/band/band250_3_antiferro.col"
file_path = "graphs/gset/G14.col"
num_spins, num_edges, edges, opt_cut, opt_energy, mu_max, ave_abs_j = parse_graph(file_path)

# Define the initial alpha values for each spin

#%% Run q-PDC model
T = 100         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 200000

poly_order = 5 # order of polynomial model (Polynomial model)
beta_schedule  = np.ones(num_steps) * (1/(1+ave_abs_j))
gamma_schedule = np.linspace(0,1.755, num_steps)
gamma_schedule *= 1

print(f"Best‐only mode = {best_only}\n")

print("Running q-PDC model...")
start = time.time()
res_qpdc = potts_sim.run_polynomial(
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
energies = np.array(res_qpdc["energy"])
cuts     = np.array(res_qpdc["cut_value"])
best_e = energies[0] if best_only else energies.min()
best_c = cuts[0]     if best_only else cuts.max()
print(f"q-PDC best energy = {best_e} (opt {opt_energy}, diff {best_e-opt_energy})")
print(f"q-PDC best cut   = {best_c} (opt {opt_cut},   diff {best_c-opt_cut})")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

if not best_only:
    # invoke plots
    plot_model(res_qpdc, "q-PDC")
    plt.show()


#%% Run NEC

T = 5e3         # total simulation time
dt = 1e-2       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 200000

alpha_rate = 1e-2 # rate of change of alpha (NEC model)
r_target = 1 # target radius for NEC model
initial_alpha_arr = -mu_max * np.ones(num_spins) # initial alpha values for NEC model

gamma_schedule  = np.linspace(0, 1, num_steps)
gamma_schedule *= 1.5 # scale by factor

print("Running NEC model...")
start = time.time()
res_nec = potts_sim.run_nec(
    T, dt, num_spins, num_states,
    edges,
    noise_factor, seed,
    alpha_rate, r_target,
    initial_alpha_arr, gamma_schedule,
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

if not best_only:
    # invoke plots
    plot_model(res_nec, "NEC")
    plt.show()

#%% Run polynomial model

T = 100         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 200000

poly_order = 20 # order of polynomial model (Polynomial model)
beta_schedule  = np.linspace(1/mu_max, 1/mu_max + 2, num_steps) # beta schedule (Polynomial and Sigmoid models)
gamma_schedule = beta_schedule * 3 # gamma schedule (Polynomial, Sigmoid, and Fixed-Amplitude models)

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

if not best_only:
    # invoke plots
    plot_model(res_poly, "Polynomial")
    plt.show()

#%% Run Sigmoid model

T = 1000         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 250000

alpha = -40.0 # alpha value for sigmoid model (Sigmoid model)

beta_schedule  = np.linspace((1-alpha)/mu_max, (1-alpha)/mu_max + 20, num_steps) # beta schedule (Polynomial and Sigmoid models)
gamma_schedule = beta_schedule * 1 # gamma schedule (Polynomial, Sigmoid, and Fixed-Amplitude models)

# repeat for Sigmoid
print("Running sigmoid model...")
start = time.time()
results = potts_sim.run_sigmoid(
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
energies = np.array(results["energy"])
cuts     = np.array(results["cut_value"])
best_e   = energies[0] if best_only else energies.min()
best_c   = cuts[0]     if best_only else cuts.max()
print(f"Sigmoid best energy    = {best_e} (opt {opt_energy}, diff {best_e-opt_energy}), pct {100*(best_e-opt_energy)/opt_energy:.2f}%")
print(f"Sigmoid best cut       = {best_c} (opt {opt_cut},   diff {best_c-opt_cut}), pct {100*(best_c-opt_cut)/opt_cut:.2f}%")
print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")

if not best_only:
    # invoke plots
    plot_model(results, "Sigmoid")
    plt.show()
else:
    # Plot distribution of best continuous state amplitudes
    cont = results["continuous_states"]
    amp = np.abs(cont)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(amp[0, :], bins=50, density=True, alpha=0.7)
    ax.set_title("Distribution of Best Continuous State Amplitudes")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")
    plt.tight_layout()
    # Print percentage of spins in the last bin
    last_bin_count = np.sum(amp[0, :] > 0.9)
    total_count = amp.shape[1]
    print(f"Percentage of spins with amplitude > 0.9: {100 * last_bin_count / total_count:.2f}%")

#%% Run Fixed-Amplitude model

T = 1000         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps
best_only = num_steps > 200000

gamma_schedule = np.linspace(0,5,num_steps)

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

if not best_only:
    # invoke plots
    plot_model(res_fix, "Fixed-Amplitude")
    plt.show()
