import numpy as np
from numba import njit
import pandas as pd
from potts_utils import parse_graph

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
def amps_to_states(X, num_vertices, num_states=3):
    '''Convert the spin amplitudes X to states in OH-encoding.'''
    states = np.zeros(num_vertices,dtype=np.int64)
    for i in range(num_vertices):
        spin_group = X[i:len(X):num_vertices] # get the num_states spins of vertex i
        assert len(spin_group) == num_states
        if sum(spin_group>0) == 1: # check that only one spin is positive (OH-encoding)
            states[i] = np.argmax(spin_group)
        else: # not correctly OH-encoded
            states[i] = -1
    return states

@njit
def num_wrong_edges_graph(states, edges):
    '''Count the number of edges that are not cut by the states. (i.e. opposite of cut value)'''
    num = 0
    for i,j in edges.T:
        if (states[i] == states[j]) or (states[i] == -1) or (states[j] == -1):
            num += 1
    return num

@njit
def check_spinflip(old_signs,X):
    new_signs = np.sign(X).astype(np.int64)
    return not np.all(old_signs == new_signs)

def get_cim_matrices(num_vertices, num_states, edges, B=18/5, zeta=0.6):
    """
    Helper function to create J and H matrices for CIM simulation.
    Constructs the coupling matrix J and the external field H for the max-3-cut problem according to Eq. 11-12 in https://arxiv.org/abs/2505.08796
    
    Parameters:
    -----------
    num_vertices : int
        Number of vertices in the graph
    num_states : int
        Number of possible states (colors)
    edges : numpy.ndarray
        Array of shape (2, num_edges) containing edges
    B : float, optional
        B/A ratio for soft constraints (default: 18/5)
    zeta : float, optional
        Empirical rescaling factor (default: 0.6)
        
    Returns:
    --------
    tuple
        (J, H) matrices for CIM simulation
    """
    
    num_spins = num_vertices * num_states

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
    
    for u,v in edges.T:
        for i in range(num_states):
            idx0 = i*num_vertices + u
            idx1 = i*num_vertices + v
            J[idx0,idx1] += (-B) * 0.25
            J[idx1,idx0] += (-B) * 0.25
            H[idx0] += (-B) * 0.25
            H[idx1] += (-B) * 0.25
    
    H *= zeta # (This is an emprical rescaling factor. It turns out you have to set this to 0.6 (as explained in https://arxiv.org/abs/2505.08796).)
    return J, H

# Main simulation function JIT-compiled as a whole
@njit
def _run_cim_njit(T, dt, num_vertices, num_states,
                 edges, noise_factor, seed,
                 alpha,
                 beta_schedule,
                 J, H,
                 return_full_history):
    """JIT-compiled implementation of CIM simulation"""
    # Setup
    np.random.seed(seed)  # This will now affect all random operations in Numba context
    num_steps = int(np.floor(T / dt))
    num_spins = num_vertices * num_states
    
    # Initialize spin amplitudes with small random values
    X = (np.random.rand(num_spins)*2-1)*1e-10
    
    # Arrays to store all timesteps - only allocate if full history is requested
    if return_full_history:
        all_continuous_states = np.zeros((num_steps, num_spins))
        all_discrete_states = np.zeros((num_steps, num_vertices), dtype=np.int64)
        all_energy = np.zeros(num_steps)
        all_cut_value = np.zeros(num_steps)
    else:
        # Use empty arrays when we don't need full history
        all_continuous_states = np.zeros((1, num_spins))  # Just a placeholder
        all_discrete_states = np.zeros((1, num_vertices), dtype=np.int64)
        all_energy = np.zeros(1)
        all_cut_value = np.zeros(1)
    
    # For tracking best solution
    best_step = 0
    best_energy = float('inf')
    best_cut_value = -float('inf')
    best_continuous_state = np.zeros(num_spins)
    best_discrete_state = np.zeros(num_vertices, dtype=np.int64)
    
    # To store last state
    last_discrete_states = np.zeros(num_vertices, dtype=np.int64)
    last_energy = 0.0
    last_cut_value = 0.0
    
    # Main simulation loop
    for step in range(num_steps):
        # Update spin amplitudes
        X = update_O12_tanh(X, alpha, beta_schedule[step], J, H, dt, noise_factor)
        
        # Calculate discrete states, energy, and cut value
        discrete_states = amps_to_states(X, num_vertices, num_states)
        num_wrong = num_wrong_edges_graph(discrete_states, edges)
        energy = float(num_wrong)
        cut_value = float(len(edges.T) - num_wrong)
        
        # Track best solution
        if energy < best_energy:
            best_energy = energy
            best_cut_value = cut_value
            best_step = step
            best_continuous_state = X.copy()
            best_discrete_state = discrete_states.copy()
        elif cut_value > best_cut_value:
            best_cut_value = cut_value
            best_energy = energy
            best_step = step
            best_continuous_state = X.copy()
            best_discrete_state = discrete_states.copy()
        
        # Update last state
        if step == num_steps - 1:
            last_discrete_states = discrete_states.copy()
            last_energy = energy
            last_cut_value = cut_value
        
        # Store states only if full history is requested
        if return_full_history:
            all_continuous_states[step] = X
            all_discrete_states[step] = discrete_states
            all_energy[step] = energy
            all_cut_value[step] = cut_value
    
    return (all_continuous_states, all_discrete_states, all_energy, all_cut_value, 
            best_continuous_state, best_discrete_state, best_energy, best_cut_value, best_step,
            X, last_discrete_states, last_energy, last_cut_value)

# Python wrapper function that handles the dictionary creation and return flags
def run_cim(T, dt, num_vertices, num_states,
            edges, noise_factor, seed,
            alpha,
            beta_schedule,
            J, H,
            return_continuous_states = True,
            return_discrete_states = False,

            return_cut_value = False,
            return_best_only=False,
            return_last_only=False):
    """
    Run CIM simulation with parameters similar to potts_sim interface.
    """
    # Validate inputs
    num_steps = int(np.floor(T / dt))
    if len(beta_schedule) != num_steps:
        raise ValueError(f"beta_schedule length ({len(beta_schedule)}) must equal num_steps ({num_steps})")
    
    # Only request full history if not using best_only or last_only
    return_full_history = not (return_best_only or return_last_only)
    
    # Run the JIT-compiled simulation
    (all_continuous_states, all_discrete_states, all_energy, all_cut_value,
     best_continuous_state, best_discrete_state, best_energy, best_cut_value, best_step,
     last_continuous_state, last_discrete_states, last_energy, last_cut_value) = \
        _run_cim_njit(T, dt, num_vertices, num_states, edges, noise_factor, seed, 
                      alpha, beta_schedule, J, H, 
                      return_full_history)
    
    # Prepare return values based on flags
    result = {}
    
    if return_best_only:
        if return_continuous_states:
            continuous_state_history = np.array([best_continuous_state])
        if return_discrete_states:
            discrete_state_history = np.array([best_discrete_state])
        if return_energy:
            energy_history = np.array([best_energy])
        if return_cut_value:
            cut_value_history = np.array([best_cut_value])
        result["step"] = best_step
        
    elif return_last_only:
        if return_continuous_states:
            continuous_state_history = np.array([last_continuous_state])
        if return_discrete_states:
            discrete_state_history = np.array([last_discrete_states])
        if return_energy:
            energy_history = np.array([last_energy])
        if return_cut_value:
            cut_value_history = np.array([last_cut_value])
        result["step"] = num_steps - 1
        
    else:
        if return_continuous_states:
            continuous_state_history = all_continuous_states
        if return_discrete_states:
            discrete_state_history = all_discrete_states
        if return_energy:
            energy_history = all_energy
        if return_cut_value:
            cut_value_history = all_cut_value
    
    # Add results to output dictionary
    result["continuous_states"] = continuous_state_history if return_continuous_states else None
    result["discrete_states"] = discrete_state_history if return_discrete_states else None
    result["energy"] = energy_history if return_energy else None
    result["cut_value"] = cut_value_history if return_cut_value else None
    result["num_spins"] = num_vertices * num_states
    
    return result

def run_cim_from_graph(T, dt, num_vertices, num_states,
                      edges,
                      noise_factor, seed,
                      alpha,
                      beta_schedule,
                      B=18/5, zeta=0.6,
                      return_continuous_states=True,
                      return_discrete_states=False,
                      return_energy=False,
                      return_cut_value=False,
                      return_best_only=False,
                      return_last_only=False):
    """
    Run CIM simulation directly from a graph, similar to potts_sim interface.
    
    Parameters:
    -----------
    T, dt, num_vertices, num_states, noise_factor, seed : same as run_cim
    edges : numpy.ndarray
        Array of shape (2, num_edges) containing source and target indices
    alpha : float
        Parameter for tanh nonlinearity
    beta_schedule : list or numpy.ndarray
        Time-dependent annealing schedule
    B : float, optional
        B/A ratio for soft constraints
    zeta : float, optional
        Empirical rescaling factor
    return_* : bool, optional
        Output options
        
    Returns:
    --------
    dict
        A dict with keys "continuous_states", "discrete_states", "energy", "cut_value", and "step"
    """
    
    # Get J and H matrices
    J, H = get_cim_matrices(num_vertices, num_states, edges, B, zeta)
    
    # Run simulation
    return run_cim(T, dt, num_vertices, num_states,
                  edges,
                  noise_factor, seed,
                  alpha,
                  beta_schedule,
                  J, H, 
                  return_continuous_states,
                  return_discrete_states,
                  return_energy,
                  return_cut_value,
                  return_best_only,
                  return_last_only)
