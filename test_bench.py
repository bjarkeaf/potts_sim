# Test bench for Potts simulation models
# This script runs various Potts simulation models and visualizes the results.

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
from potts_utils import parse_graph, compute_largest_eigenvalues_and_eigenvectors, compute_max_eigenvalue
from cim_sim import run_cim_from_graph

#%% Define helper functions

def plot_phase_with_wraparound(ax, t, phase, **kwargs):
    """Plot phase data with disconnected lines at phase wrapping points."""
    # Ensure phase is 2D array (time, spins) to handle single-spin input
    if phase.ndim == 1:
        phase = phase[:, np.newaxis]
    for i in range(min(5, phase.shape[1])):
        t_plot = np.copy(t)
        phase_plot = np.copy(phase[:, i])

        # Find jumps larger than threshold (π)
        jumps = np.where(np.abs(np.diff(phase_plot)) > np.pi)[0]

        # Insert wrap points, NaN, and opposite wrap at each jump
        for idx in jumps[::-1]:  # go backwards to keep indices valid
            t_cross = t_plot[idx + 1]
            delta = phase_plot[idx + 1] - phase_plot[idx]
            if delta > 0:
                # crossed from -π to +π
                b0, b1 = -np.pi, np.pi
            else:
                # crossed from +π to -π
                b0, b1 = np.pi, -np.pi

            # insert [boundary, NaN, opposite boundary] at crossing time
            t_plot = np.insert(t_plot, idx + 1, [t_cross, t_cross, t_cross])
            phase_plot = np.insert(phase_plot, idx + 1, [b0, np.nan, b1])

        ax.plot(t_plot, phase_plot, **kwargs)

def plot_model(res, name, dt, num_steps, num_vertices, num_states, opt_cut=0):
        cont   = res["continuous_states"]
        disc   = res["discrete_states"]
        cut    = res["cut_value"]
        num_spins = res.get("num_spins", num_vertices)
        t_full = np.arange(num_steps) * dt

        # downsample for plotting
        if cont.shape[0] > 1000:
            idxs = np.linspace(0, cont.shape[0] - 1, 1000, dtype=int)
        else:
            idxs = np.arange(cont.shape[0])
        t   = t_full[idxs]
        cont= cont[idxs]
        disc= disc[idxs]
        cut    = cut[idxs]

        if np.iscomplexobj(cont):
            # If continuous states are complex, plot amplitude and phase
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
            amp   = np.abs(cont)
            phase = np.angle(cont)
            for i in range(num_spins):
                ax1.plot(t, amp[:,i],   lw=0.8)
                #ax2.plot(t, phase[:,i], lw=0.8)
                plot_phase_with_wraparound(ax2, t, phase[:,i], lw=0.8)
            # add reference phase lines
            for j in range(num_states):
                theta = 2*np.pi/num_states * (j - np.floor(num_states/2))
                ax2.axhline(theta, color='gray', linestyle='--', lw=0.5)
            ax1.set_ylabel("Amplitude")
            ax2.set_ylabel("Phase")
            ax2.set_xlabel("Time")
            fig.suptitle(f"{name} Continuous States")
            plt.tight_layout()
        else:
            # If continuous states are real, just plot them directly
            amp = cont
            fig, ax = plt.subplots(figsize=(5,5))
            for i in range(num_spins):
                ax.plot(t, amp[:,i], lw=0.8)
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("Time")
            
        # discrete states
        fig, ax = plt.subplots(figsize=(10,4))
        for i in range(num_vertices):
            ax.plot(t, disc[:,i], lw=0.8)
        ax.set_ylabel("Discrete State")
        ax.set_xlabel("Time")
        fig.suptitle(f"{name} Discrete States")
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

def plot_histograms(res, name, num_states):
    # Plot distribution of last continuous state amplitudes and phases
    cont = res["continuous_states"]
    last_cont = cont[-1]         # shape (num_spins,)
    if np.iscomplexobj(last_cont):
        # If last continuous state is complex, separate amplitude and phase
        amp       = np.abs(last_cont)
        phase     = np.angle(last_cont)
    else:
        # If last continuous state is real, just use it as amplitude
        amp       = last_cont

    # Create side-by-side histograms for amplitude & phase
    if np.iscomplexobj(last_cont):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = None

    # Amplitude histogram
    if np.ptp(amp) > 1e-6:
        # If amplitude has significant range (not homogeneous), plot its distribution
        n_amp_bins = min(50, np.unique(amp).size) if amp.size > 0 else 1
        ax1.hist(amp, bins=n_amp_bins, density=True, alpha=0.7)
        ax1.set_title("Distribution of Final Continuous State Amplitudes")
        ax1.set_xlabel("Amplitude")
        ax1.set_ylabel("Density")

    # Phase histogram
    if np.iscomplexobj(last_cont):
        # If last continuous state is complex, plot phase distribution
        n_phase_bins = min(50, np.unique(phase).size) if phase.size > 0 else 1
        ax2.hist(phase, bins=n_phase_bins, density=True, alpha=0.7)
        ax2.set_title("Distribution of Final Continuous State Phases")
        ax2.set_xlabel("Phase (radians)")
        ax2.set_ylabel("Density")

    plt.tight_layout()

    # Print percentage of spins with amplitude > 0.9
    pct_high_amp = 100 * np.sum(amp > 0.9) / amp.size
    print(f"Percentage of spins with amplitude > 0.9: {pct_high_amp:.2f}%")

    # Plot histogram of last discrete states
    last_disc = res["discrete_states"][-1]  # shape (num_spins,)
    fig, ax3 = plt.subplots(figsize=(6, 4))
    if np.any(last_disc == -1):
        # Count states (including -1 meaning improperly encoded)
        all_states = np.arange(-1, num_states)
    else:
        all_states = np.arange(num_states)
    counts = [np.sum(last_disc == state) for state in all_states]
    ax3.bar(all_states, counts, align='center', alpha=0.7)
    ax3.set_xticks(all_states)
    ax3.set_xlabel("Discrete State")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Final Discrete States")
    plt.tight_layout()

def execute_model(name, run_func, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed, *extra_args):
    num_steps = int(np.floor(T / dt))
    last_only = num_steps > 2e5
    if name == "CIM":
        last_only = num_steps > 1e5
    if last_only:
        print(f"Running {name} model in last-only mode (only last state will be returned)...")
    else:
        print(f"Running {name} model in full mode (all states will be returned)...")
    kwargs = dict(
        return_continuous_states=True,
        return_discrete_states=True,
        return_cut_value=True,
        return_best_only=False,
        return_last_only=last_only
    )
    start = time.time()
    res = run_func(
        T, dt,
        num_vertices, num_states,
        edges,
        noise_factor, seed,
        *extra_args,
        **kwargs
    )
    duration = time.time() - start
    cuts = np.array(res["cut_value"])
    last_c = cuts[0] if last_only else cuts.max()
    pct = 100 * (last_c - opt_cut) / opt_cut
    print(f"{name} best cut = {last_c} (opt {opt_cut}, diff {last_c-opt_cut}), pct {pct:.2f}%")
    print(f"Time per step: {duration*1e6/num_steps:.2f} μs\n")
    if not last_only:
        plot_model(res, name, dt, num_steps, num_vertices, num_states, opt_cut)
        plt.show()
    plot_histograms(res, name, num_states)
    res["dt"] = dt
    res["num_steps"] = num_steps
    return res

#%% Set up the simulation parameters
if __name__ == "__main__":
    # Default parameters
    T = 100         # total simulation time
    dt = 1e-3       # time step

    num_states = 4 # number of states for Potts model (q=k)

    noise_factor = 1e-4
    seed = 2

    # Load a coupling graph
    #file_path = "graphs/band/band50_3_antiferro.col"
    file_path = "graphs/gset/G5.col"
    #file_path = "graphs/g05/g05_10.0.col"
    num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max = parse_graph(file_path)
    opt_cut = opt_cut_dict.get(num_states, 0)
    
    edges_per_vertex = num_edges / num_vertices  # average number of edges per vertex

    res_dict = {}  # Dictionary to store results for each model

   #%% Run polynomial model
    T = 1000         # total simulation time
    dt = 1e-3       # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    poly_order = 11 # order of polynomial model (Polynomial model)

    beta_schedule = np.linspace(1/mu_max, 1/mu_max + 2, num_steps) # beta schedule
    gamma_schedule = beta_schedule * 0.25 # gamma schedule

    amplitude_seed = 0.0

    res_dict["Polynomial"] = execute_model("Polynomial", potts_sim.run_polynomial, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        amplitude_seed,
        poly_order,
        list(beta_schedule), list(gamma_schedule)
    )

    #%% Run q-PDC model
    T = 50         # total simulation time
    dt = 1e-3       # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    dampening = 1 + edges_per_vertex
    gamma_th = (256/27)**(1/4)

    poly_order = 2*num_states - 1 # order of polynomial model (Polynomial model)
    beta_schedule  = np.ones(num_steps) * 1 / dampening # constant beta schedule
    gamma_schedule = np.linspace(0,gamma_th, num_steps)
    gamma_schedule *= 32

    amplitude_seed = 1

    res_dict["q-PDC"] = execute_model("q-PDC", potts_sim.run_polynomial, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        amplitude_seed,
        poly_order,
        list(beta_schedule), list(gamma_schedule)
    )

    #%% Run NEC

    T = 2e3         # total simulation time
    dt = 1e-2       # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    polynomial_order = 4 # n=3 used in original paper, but n>=q is required for amplitude bounding
    alpha_rate = 1e-2 # rate of change of alpha (NEC model)
    r_target = 1 # target radius for NEC model
    initial_alpha_arr = -mu_max * np.ones(num_vertices) # initial alpha values for NEC model

    # constant gamma
    #gamma_schedule = np.ones(num_steps) * 0.3

    # linear gamma schedule
    gamma_schedule  = np.linspace(0, 1, num_steps)
    gamma_schedule *= 2.5 # scale by factor

    res_dict["NEC"] = execute_model("NEC", potts_sim.run_nec, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        polynomial_order, alpha_rate, r_target,
        initial_alpha_arr, gamma_schedule
    )

    #%% Run Sigmoid model

    T = 2000         # total simulation time
    dt = 1e-3       # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    alpha = -50.0 # alpha value for sigmoid model

    beta_schedule  = np.linspace((1-alpha)/mu_max, (1-alpha)/mu_max + 20, num_steps) # beta schedule (Polynomial and Sigmoid models)
    gamma_schedule = beta_schedule * 4 # gamma schedule (Polynomial, Sigmoid, and Fixed-Amplitude models)

    res_dict["Sigmoid"] = execute_model("Sigmoid", potts_sim.run_sigmoid, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        alpha, list(beta_schedule), list(gamma_schedule)
    )

    #%% Run Fixed-Amplitude model

    T = 100         # total simulation time
    dt = 1e-3       # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    gamma_schedule = np.linspace(0,10,num_steps)

    res_dict["Fixed-Amplitude"] = execute_model("Fixed-Amplitude", potts_sim.run_fixed_amplitude, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        list(gamma_schedule)
    )

    #%% Run CIM model

    T = 1e3          # total simulation time
    dt = 1e-2        # time step
    num_steps = int(np.floor(T / dt))  # number of time steps

    # CIM specific parameters
    zeta = 0.7       # empirical rescaling factor
    B_num_vertices = 225  # B/A/num_    vertices ratio for soft constraints 
    B = B_num_vertices / num_vertices  # B value for CIM model
    alpha = -50      # parameter for tanh nonlinearity
    beta_schedule = np.linspace(0, 0.02, num_steps)  # time-dependent annealing schedule

    # Execute the CIM model
    res_dict["CIM"] = execute_model("CIM", run_cim_from_graph, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
        alpha,
        list(beta_schedule),
        B, zeta
    )

# %%
