#!/usr/bin/env python3
import numpy as np
import os
import sys
import time
import yaml
import argparse
import functools
import itertools
import hashlib
import pandas as pd
import glob
from enum import Enum
from datetime import datetime
from mpi4py import MPI
from pathlib import Path

import potts_sim
from graph_parser import parse_graph

class ModelType(Enum):
    """Enum for the different Potts model types"""
    NEC = "nec"
    POLYNOMIAL = "polynomial" 
    SIGMOID = "sigmoid"
    FIXED_AMPLITUDE = "fixed_amplitude"

def get_git_revision():
    """Get git commit hash for reproducibility"""
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       stderr=subprocess.PIPE).decode('ascii').strip()
    except:
        return "unknown"

@functools.lru_cache(maxsize=32)
def load_graph(graph_path):
    """Parse a graph file, cached to avoid repeated parsing"""
    num_spins, num_edges, edges, opt_cut, opt_energy = parse_graph(graph_path)
    return {
        'path': graph_path,
        'name': Path(graph_path).stem,
        'num_spins': num_spins,
        'num_edges': num_edges,
        'edges': edges,
        'opt_cut': opt_cut,
        'opt_energy': opt_energy,
    }

def parse_schedule(schedule_str, num_steps):
    """Parse schedule string (e.g., 'lin(0,1)') into numpy array"""
    if schedule_str.startswith("lin(") and schedule_str.endswith(")"):
        # Linear schedule: lin(start,end)
        params = schedule_str[4:-1].split(",")
        start, end = float(params[0]), float(params[1])
        return np.linspace(start, end, num_steps).tolist()
    elif schedule_str.startswith("exp(") and schedule_str.endswith(")"):
        # Exponential schedule: exp(start,end,factor)
        params = schedule_str[4:-1].split(",")
        start, end, factor = float(params[0]), float(params[1]), float(params[2])
        base = np.linspace(0, 1, num_steps)
        exp_vals = (np.exp(factor * base) - 1) / (np.exp(factor) - 1)
        return (start + (end - start) * exp_vals).tolist()
    else:
        raise ValueError(f"Unknown schedule format: {schedule_str}")

def generate_param_sets(model_type, model_params, T, dt):
    """Generate all parameter combinations for a model type"""
    num_steps = int(np.floor(T / dt))
    
    # For each model type, define the parameters to sweep
    # and how to generate the full parameter set
    if model_type == ModelType.POLYNOMIAL:
        poly_orders = model_params.get('poly_order', [3])
        beta_schedules = model_params.get('beta_schedule', ["lin(0,1)"])
        gamma_schedules = model_params.get('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        for poly_order, beta_str, gamma_str in itertools.product(poly_orders, beta_schedules, gamma_schedules):
            # Generate actual schedule arrays
            beta_schedule = parse_schedule(beta_str, num_steps)
            gamma_schedule = parse_schedule(gamma_str, num_steps)
            
            # Create a unique identifier for this parameter set
            param_id = f"po{poly_order}_b{beta_str}_g{gamma_str}"
            
            param_sets.append({
                'id': param_id,
                'poly_order': poly_order,
                'beta_schedule': beta_schedule,
                'gamma_schedule': gamma_schedule,
                'beta_str': beta_str,
                'gamma_str': gamma_str
            })
        return param_sets
        
    elif model_type == ModelType.NEC:
        alpha_rates = model_params.get('alpha_rate', [1e-2])
        gammas = model_params.get('gamma', [1.0])
        r_targets = model_params.get('r_target', [2.0])
        initial_alphas = model_params.get('initial_alpha', [1.0])
        
        param_sets = []
        for alpha_rate, gamma, r_target, initial_alpha in itertools.product(
                alpha_rates, gammas, r_targets, initial_alphas):
            # Create a unique identifier for this parameter set
            param_id = f"ar{alpha_rate:.2e}_g{gamma}_rt{r_target}_ia{initial_alpha}"
            
            param_sets.append({
                'id': param_id,
                'alpha_rate': alpha_rate,
                'gamma': gamma,
                'r_target': r_target,
                'initial_alpha': initial_alpha
            })
        return param_sets
        
    elif model_type == ModelType.SIGMOID:
        alphas = model_params.get('alpha', [-1.0])
        beta_schedules = model_params.get('beta_schedule', ["lin(0,1)"])
        gamma_schedules = model_params.get('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        for alpha, beta_str, gamma_str in itertools.product(alphas, beta_schedules, gamma_schedules):
            # Generate actual schedule arrays
            beta_schedule = parse_schedule(beta_str, num_steps)
            gamma_schedule = parse_schedule(gamma_str, num_steps)
            
            # Create a unique identifier for this parameter set
            param_id = f"a{alpha}_b{beta_str}_g{gamma_str}"
            
            param_sets.append({
                'id': param_id,
                'alpha': alpha,
                'beta_schedule': beta_schedule,
                'gamma_schedule': gamma_schedule,
                'beta_str': beta_str,
                'gamma_str': gamma_str
            })
        return param_sets
        
    elif model_type == ModelType.FIXED_AMPLITUDE:
        gamma_schedules = model_params.get('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        for gamma_str in gamma_schedules:
            # Generate actual schedule array
            gamma_schedule = parse_schedule(gamma_str, num_steps)
            
            # Create a unique identifier for this parameter set
            param_id = f"g{gamma_str}"
            
            param_sets.append({
                'id': param_id,
                'gamma_schedule': gamma_schedule,
                'gamma_str': gamma_str
            })
        return param_sets
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_task(graph, model_type, param_set, seed, T, dt, num_states, noise_factor):
    """Run a single task with the given parameters"""
    num_spins = graph['num_spins']
    edges = graph['edges']
    start_time = time.time()
    
    # Generate a deterministic seed that's unique for this task
    task_seed = int(hashlib.md5(f"{graph['path']}_{model_type.name}_{param_set['id']}_{seed}".encode()).hexdigest(), 16) % (2**31)
    
    # Run the appropriate model
    if model_type == ModelType.POLYNOMIAL:
        # For POLYNOMIAL model
        res = potts_sim.run_polynomial(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['poly_order'],
            param_set['beta_schedule'], param_set['gamma_schedule'],
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    elif model_type == ModelType.NEC:
        # For NEC model
        initial_alpha_arr = [param_set['initial_alpha']] * num_spins
        res = potts_sim.run_nec(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['alpha_rate'], param_set['gamma'], param_set['r_target'],
            initial_alpha_arr,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    elif model_type == ModelType.SIGMOID:
        # For SIGMOID model
        res = potts_sim.run_sigmoid(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['alpha'],
            param_set['beta_schedule'], param_set['gamma_schedule'],
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    elif model_type == ModelType.FIXED_AMPLITUDE:
        # For FIXED_AMPLITUDE model
        res = potts_sim.run_fixed_amplitude(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['gamma_schedule'],
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract results
    best_cut = float(res["cut_value"][0])
    best_energy = float(res["energy"][0])
    best_step = int(res["step"])
    elapsed_time = time.time() - start_time
    
    # Return results in a dictionary
    result = {
        "graph": graph['name'],
        "model": model_type.name,
        "param_id": param_set['id'],
        "seed": seed,
        "cut_value": best_cut,
        "energy": best_energy,
        "step": best_step,
        "opt_cut": graph['opt_cut'],
        "opt_energy": graph['opt_energy'],
        "cut_diff": best_cut - graph['opt_cut'] if graph['opt_cut'] is not None else None,
        "energy_diff": best_energy - graph['opt_energy'] if graph['opt_energy'] is not None else None,
        "runtime": elapsed_time
    }
    
    # Add model-specific parameters to the result
    if model_type == ModelType.POLYNOMIAL:
        result.update({
            "poly_order": param_set['poly_order'],
            "beta_schedule": param_set['beta_str'],
            "gamma_schedule": param_set['gamma_str']
        })
    elif model_type == ModelType.NEC:
        result.update({
            "alpha_rate": param_set['alpha_rate'],
            "gamma": param_set['gamma'],
            "r_target": param_set['r_target'],
            "initial_alpha": param_set['initial_alpha']
        })
    elif model_type == ModelType.SIGMOID:
        result.update({
            "alpha": param_set['alpha'],
            "beta_schedule": param_set['beta_str'],
            "gamma_schedule": param_set['gamma_str']
        })
    elif model_type == ModelType.FIXED_AMPLITUDE:
        result.update({
            "gamma_schedule": param_set['gamma_str']
        })
    
    return result

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 parses command line arguments
    if rank == 0:
        parser = argparse.ArgumentParser(description='Run Potts model parameter sweep')
        parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
        args = parser.parse_args()
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract general parameters
        num_runs = config.get('num_runs', 100)
        graph_path = config.get('graph_path', 'graphs/')
        out_dir = config.get('out_dir', 'results/')
        T = config.get('T', 1000.0)
        dt = config.get('dt', 1e-3)
        num_states = config.get('num_states', 3)
        noise_factor = config.get('noise_factor', 1e-4)
        
        # Make sure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Identify graph files
        if os.path.isdir(graph_path):
            graph_files = glob.glob(os.path.join(graph_path, "*.col"))
        else:
            graph_files = [graph_path]
        
        if not graph_files:
            print(f"Error: No .col files found in {graph_path}")
            comm.Abort(1)
            
        print(f"Found {len(graph_files)} graph files")
        
        # Build task list
        tasks = []
        for graph_file in graph_files:
            graph = load_graph(graph_file)
            
            # Process each model type
            for model_name, model_params in config.get('models', {}).items():
                try:
                    model_type = ModelType[model_name]
                except KeyError:
                    print(f"Warning: Unknown model type {model_name}, skipping")
                    continue
                
                # Generate all parameter sets for this model
                param_sets = generate_param_sets(model_type, model_params, T, dt)
                
                # For each parameter set, add tasks for all seeds
                for param_set in param_sets:
                    for seed in range(num_runs):
                        tasks.append({
                            'graph': graph,
                            'model_type': model_type,
                            'param_set': param_set,
                            'seed': seed,
                            'T': T,
                            'dt': dt,
                            'num_states': num_states,
                            'noise_factor': noise_factor
                        })
        
        print(f"Generated {len(tasks)} tasks")
        
        # Save run configuration
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'git_revision': get_git_revision(),
            'config_file': args.config,
            'num_tasks': len(tasks),
            'num_graphs': len(graph_files),
            'num_ranks': size
        }
        
        with open(os.path.join(out_dir, 'run_info.yaml'), 'w') as f:
            yaml.dump(run_info, f)
    else:
        # Non-root ranks initialize these variables
        tasks = None
        out_dir = None
        run_info = None
    
    # Broadcast task list and output directory to all ranks
    tasks = comm.bcast(tasks, root=0)
    out_dir = comm.bcast(out_dir, root=0)
    
    # Divide tasks among ranks using simple strided allocation
    my_tasks = tasks[rank::size]
    print(f"Rank {rank}: Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    local_results = []
    for i, task in enumerate(my_tasks):
        try:
            result = run_task(**task)
            local_results.append(result)
            
            if i % 10 == 0:
                print(f"Rank {rank}: Completed {i+1}/{len(my_tasks)} tasks")
        except Exception as e:
            print(f"Rank {rank}: Error processing task {i}: {e}")
    
    # Convert results to DataFrame
    if local_results:
        local_df = pd.DataFrame(local_results)
    else:
        # Create empty DataFrame with expected columns if no results
        local_df = pd.DataFrame()
    
    # Store local results to disk (useful for recovery if job fails)
    local_out = os.path.join(out_dir, f"results_rank{rank}.parquet")
    if not local_df.empty:
        local_df.to_parquet(local_out, index=False)
    
    # Gather all DataFrames at rank 0
    all_dfs = comm.gather(local_df, root=0)
    
    # Rank 0 combines and saves the final results
    if rank == 0:
        # Filter out empty DataFrames and concatenate
        all_dfs = [df for df in all_dfs if not df.empty]
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save the combined results
            combined_out = os.path.join(out_dir, "results.parquet")
            combined_df.to_parquet(combined_out, index=False)
            
            print(f"Saved combined results with {len(combined_df)} rows to {combined_out}")
        else:
            print("Warning: No results were collected from any rank")

if __name__ == "__main__":
    main()
