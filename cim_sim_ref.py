import numpy as np
import os
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt
from potts_utils import parse_graph

# Problem specification
problem_path = 'graphs/g05/g05_20.0.col'
problem_name = os.path.basename(problem_path).rsplit('.', 1)[0]


# Basic simulation parameters
num_states = 3 # This code only works for max-3-cut at the moment.
noise_factor = 1e-4
seed = 2  # Random seed for reproducibility
num_inits = 1  # Number of initializations (1 for visualization)

T = 1e3          # total simulation time
dt = 1e-2        # time step
num_steps = int(np.floor(T / dt))  # number of time steps

# CIM specific parameters
zeta = 0.6       # empirical rescaling factor
B = 18/5         # B/A ratio for soft constraints
alpha = -10.0    # parameter for tanh nonlinearity
beta_range = np.linspace(0, 0.01, num_steps)  # Annealing schedule

# Calculated parameters
max_time = T  # For backwards compatibility
annealing_speed = (beta_range[-1] - beta_range[0]) / max_time  # For reporting

# Historical parameters (kept for compatibility)
zeeman_trick = 'spinsign'

# FUNCTIONS
@njit
def update_O12_tanh(X, alpha, beta, J, H, dt, noise_factor):
    sign_X = np.sign(X)
    dXdt = -X + np.tanh(alpha * X + beta * (np.dot(J,sign_X) + H))
    X += dXdt * dt
    if noise_factor != 0:
        X += np.random.normal(0,np.sqrt(dt),size=len(X)) * noise_factor # noise
    return X

@njit
def amps_to_states(X, num_vertices):
    '''Convert the spin amplitudes X to states in OH-encoding.'''
    states = np.zeros(num_vertices,dtype=np.int64)
    for i in range(num_vertices):
        triple = X[i:len(X):num_vertices] # get the 3 spins of vertex i
        assert len(triple) == 3
        if sum(triple>0) == 1: # check that only one spin is positive (OH-encoding)
            states[i] = np.argmax(triple)
        else: # not correctly OH-encoded
            states[i] = -1
    return states

@njit
def num_wrong_edges_graph(states,edges):
    '''Count the number of edges that are not cut by the states. (i.e. opposite of cut value)'''
    num = 0
    for i,j in edges:
        if (states[i] == states[j]) or (states[i] == -1) or (states[j] == -1):
            num += 1
    return num

def get_coupling_matrix_NEW(num_vertices, num_states, num_spins, edges, B, zeta):
    """Construct the coupling matrix J and the external field H for the max-3-cut problem according to Eq. 11-12 in https://arxiv.org/abs/2505.08796."""
    A = 1. # We're scanning the ratio B/A (the prefactors of the soft constraints. See https://arxiv.org/abs/2505.08796 eq.11-12), so we can set A=1.
    
    J = np.zeros((num_spins,num_spins),dtype=np.float64)
    H = np.zeros(num_spins,dtype=np.float64)
    
    for v in range(num_vertices):
        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    idx0 = i*num_vertices + v
                    idx1 = j*num_vertices + v
                    J[idx0,idx1] += (-A) * 0.25
                    J[idx1,idx0] += (-A) * 0.25
        for i in range(num_states):
            idx = i*num_vertices + v
            H[idx] = (-A) * (num_states/2-1)
    
    for u,v in edges:
        for i in range(num_states):
            idx0 = i*num_vertices + u
            idx1 = i*num_vertices + v
            J[idx0,idx1] += (-B) * 0.25
            J[idx1,idx0] += (-B) * 0.25
            H[idx0] += (-B) * 0.25
            H[idx1] += (-B) * 0.25
    
    H *= zeta # (This is an emprical rescaling factor. It turns out you have to set this to 0.6 (as explained in https://arxiv.org/abs/2505.08796).)
    return J, H

@njit
def check_spinflip(old_signs,X):
    new_signs = np.sign(X).astype(np.int64)
    return not np.all(old_signs == new_signs)

@njit
def euler_integration(beta_range, num_inits, alpha, J, H, dt, noise_factor, GS_energy, num_spins, arr_edges, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
        
    number_of_successes = 0.
    for _ in range(num_inits):
        X = (np.random.rand(num_spins)*2-1)*10**(-10) # random small init
        old_signs = np.sign(X.copy()).astype(np.int64)
        for idx_beta, beta in enumerate(beta_range):
            X = update_O12_tanh(X, alpha, beta, J, H, dt, noise_factor)

            # Only calculate binary energy if the spin signs have changed (to save time).
            if check_spinflip(old_signs,X):
                old_signs = np.sign(X.copy()).astype(np.int64)

                states = amps_to_states(X,num_vertices)
                num_wrong_edges = num_wrong_edges_graph(states,arr_edges)
                if int(num_wrong_edges) == int(GS_energy):
                    number_of_successes += 1
                    break
                else:
                    assert int(num_wrong_edges) > int(GS_energy)
    return number_of_successes / np.float64(num_inits)

@njit
def euler_integration_with_history(beta_range, num_inits, alpha, J, H, dt, noise_factor, GS_energy, num_spins, arr_edges, num_vertices, seed=None):
    """Modified version of euler_integration that tracks the history of states for plotting."""
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility
        
    number_of_successes = 0.
    time_steps = len(beta_range)
    
    # Arrays to store history (for the first initialization only)
    all_continuous_states = np.zeros((time_steps, num_spins))
    all_discrete_states = np.zeros((time_steps, num_vertices), dtype=np.int64)
    all_cut_values = np.zeros(time_steps)
    
    for init in range(num_inits):
        X = (np.random.rand(num_spins)*2-1)*10**(-10) # random small init
        old_signs = np.sign(X.copy()).astype(np.int64)
        
        for idx_beta, beta in enumerate(beta_range):
            X = update_O12_tanh(X, alpha, beta, J, H, dt, noise_factor)
            
            # Calculate states and energy at each step (for plotting)
            states = amps_to_states(X, num_vertices)
            num_wrong_edges = num_wrong_edges_graph(states, arr_edges)
            cut_value = len(arr_edges) - num_wrong_edges
            
            # Only store history for the first initialization
            if init == 0:
                all_continuous_states[idx_beta] = X
                all_discrete_states[idx_beta] = states
                all_cut_values[idx_beta] = cut_value
            
            # Check if we found the optimal solution
            if int(num_wrong_edges) == int(GS_energy):
                number_of_successes += 1
                #break
    
    success_rate = number_of_successes / np.float64(num_inits)
    return success_rate, all_continuous_states, all_discrete_states, all_cut_values

def plot_continuous_states(continuous_states, num_vertices, num_states):
    """Plot the history of continuous states (amplitudes and phases if complex)."""
    # Downsample for plotting if needed
    if continuous_states.shape[0] > 1000:
        idxs = np.linspace(0, continuous_states.shape[0] - 1, 1000, dtype=int)
        cont = continuous_states[idxs]
        t = np.arange(len(idxs))
    else:
        cont = continuous_states
        t = np.arange(cont.shape[0])

    if np.iscomplexobj(cont):
        # If continuous states are complex, plot amplitude and phase
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        amp = np.abs(cont)
        phase = np.angle(cont)
        for i in range(cont.shape[1]):
            ax1.plot(t, amp[:, i], lw=0.8)
            ax2.plot(t, phase[:, i], lw=0.8)
        # Add reference phase lines
        for j in range(num_states):
            theta = 2*np.pi/num_states * (j - np.floor(num_states/2))
            ax2.axhline(theta, color='gray', linestyle='--', lw=0.5)
        ax1.set_ylabel("Amplitude")
        ax2.set_ylabel("Phase")
        ax2.set_xlabel("Time Steps")
        fig.suptitle("CIM Reference - Continuous States")
    else:
        # If continuous states are real, just plot them directly
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(cont.shape[1]):
            ax.plot(t, cont[:, i], lw=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time Steps")
        fig.suptitle("CIM Reference - Continuous States")
    
    plt.tight_layout()
    plt.show()

def plot_discrete_states(discrete_states):
    """Plot the history of discrete states."""
    if discrete_states.shape[0] > 1000:
        idxs = np.linspace(0, discrete_states.shape[0] - 1, 1000, dtype=int)
        disc = discrete_states[idxs]
        t = np.arange(len(idxs))
    else:
        disc = discrete_states
        t = np.arange(disc.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(disc.shape[1]):
        ax.plot(t, disc[:, i], lw=0.8)
    ax.set_ylabel("Discrete State")
    ax.set_xlabel("Time Steps")
    fig.suptitle("CIM Reference - Discrete States")
    plt.tight_layout()
    plt.show()

def plot_cut_values(cut_values, opt_cut=None):
    """Plot the history of cut values."""
    if cut_values.shape[0] > 1000:
        idxs = np.linspace(0, cut_values.shape[0] - 1, 1000, dtype=int)
        cuts = cut_values[idxs]
        t = np.arange(len(idxs))
    else:
        cuts = cut_values
        t = np.arange(cuts.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, cuts)
    if opt_cut is not None:
        ax.axhline(opt_cut, color='red', linestyle='--', label='Optimum Cut Value')
        ax.legend()
    ax.set_ylabel("Cut Value")
    ax.set_xlabel("Time Steps")
    fig.suptitle("CIM Reference - Cut Value")
    plt.tight_layout()
    plt.show()

def plot_histograms(continuous_states, discrete_states, num_states):
    """Plot histograms of final continuous state amplitudes/phases and discrete states."""
    # Final continuous state
    last_cont = continuous_states[-1]
    
    # Create histograms
    if np.iscomplexobj(last_cont):
        # Complex states: plot amplitude and phase
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
        
        amp = np.abs(last_cont)
        phase = np.angle(last_cont)
        
        n_amp_bins = min(50, np.unique(amp).size) if amp.size > 0 else 1
        ax1.hist(amp, bins=n_amp_bins, density=True, alpha=0.7)
        ax1.set_title("Distribution of Final Continuous State Amplitudes")
        ax1.set_xlabel("Amplitude")
        ax1.set_ylabel("Density")
        
        n_phase_bins = min(50, np.unique(phase).size) if phase.size > 0 else 1
        ax2.hist(phase, bins=n_phase_bins, density=True, alpha=0.7)
        ax2.set_title("Distribution of Final Continuous State Phases")
        ax2.set_xlabel("Phase (radians)")
        ax2.set_ylabel("Density")
    else:
        # Real states: plot just amplitudes
        fig, ax1 = plt.subplots(figsize=(7, 4))
        
        amp = last_cont
        n_amp_bins = min(50, np.unique(amp).size) if amp.size > 0 else 1
        ax1.hist(amp, bins=n_amp_bins, density=True, alpha=0.7)
        ax1.set_title("Distribution of Final Continuous State Amplitudes")
        ax1.set_xlabel("Amplitude")
        ax1.set_ylabel("Density")
    
    plt.tight_layout()
    plt.show()
    
    # Print percentage of spins with amplitude > 0.9
    pct_high_amp = 100 * np.sum(np.abs(last_cont) > 0.9) / last_cont.size
    print(f"Percentage of spins with amplitude > 0.9: {pct_high_amp:.2f}%")
    
    # Plot histogram of discrete states
    last_disc = discrete_states[-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if np.any(last_disc == -1):
        # Count states (including -1 meaning improperly encoded)
        all_states = np.arange(-1, num_states)
    else:
        all_states = np.arange(num_states)
    
    counts = [np.sum(last_disc == state) for state in all_states]
    ax.bar(all_states, counts, align='center', alpha=0.7)
    ax.set_xticks(all_states)
    ax.set_xlabel("Discrete State")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Final Discrete States")
    plt.tight_layout()
    plt.show()

# LOAD PROBLEM INSTANCE
dir_graphs = "cim_graphs"
#edges, weights = read_BiqMac(problem_name, dir_graphs)
num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max, ave_abs_J = parse_graph(problem_path)
edges = edges.T # transpose to get edges in (source, target) format

#num_vertices = len(np.unique(np.array(edges).ravel())) # count the number of unique vertex indices
num_spins = num_states * num_vertices # every graph vertex is represented by 3 spins (OH-encoding)

#GS_energy = load_ground_state(problem_name, len(edges), dir_graphs)
GS_energy = opt_energy_dict.get(num_states, None)
if GS_energy is None:
    raise ValueError(f"Ground state energy not found for problem {problem_name} with {num_states} colors.")
J, H = get_coupling_matrix_NEW(num_vertices, num_states, num_spins, edges, B, zeta)

# RUN SIMULATION WITH HISTORY TRACKING
time_range = np.arange(0, max_time+dt, dt)
#beta_range = annealing_speed * time_range
arr_edges = np.array(edges, dtype=np.int64)

# Run simulation and get history with seed
success_rate, continuous_states, discrete_states, cut_values = euler_integration_with_history(
    beta_range, num_inits, alpha, J, H, dt, noise_factor, GS_energy, num_spins, arr_edges, num_vertices, seed
)

# Calculate optimal cut value
opt_cut = len(edges) - GS_energy  # Calculate optimal cut value from GS_energy

# Find the best cut value achieved
best_cut = cut_values.max()
abs_diff = best_cut - opt_cut
rel_diff_pct = 100 * abs_diff / opt_cut

# Print best cut and comparison with optimum
print(f"CIM Reference best cut = {best_cut} (opt {opt_cut}, diff {abs_diff}), pct {rel_diff_pct:.2f}%")

# PRINT RESULTS
print_list = [success_rate, num_inits, zeeman_trick, problem_name, B, alpha, max_time, annealing_speed, dt, noise_factor, zeta]
print(';'.join([str(x) for x in print_list]))

# Add seed information to output
print(f"Seed used: {seed}")

# PLOT RESULTS
print("\nGenerating plots for visual comparison with test_bench.py output...")

# Plot continuous states
plot_continuous_states(continuous_states, num_vertices, num_states)

# Plot discrete states
plot_discrete_states(discrete_states)

# Plot cut values
plot_cut_values(cut_values, opt_cut)

# Plot histograms of final states
plot_histograms(continuous_states, discrete_states, num_states)

print("\nPlotting complete. Compare these results with those from test_bench.py")
