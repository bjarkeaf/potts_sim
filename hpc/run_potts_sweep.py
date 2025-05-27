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
import re
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

# Constants for simulation and performance estimation
ASSUMED_TIME_PER_STEP_US = 100  # Assumed processing time per step in microseconds

try:
    from mpi4py import MPI
    use_mpi = True
except ImportError:
    # Fallback to a dummy MPI-like interface if mpi4py is not available (local run)
    use_mpi = False
    class DummyComm:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, data, root=0): return data
        def gather(self, data, root=0): return [data]
        def Abort(self, errorcode=0): 
            print(f"Aborting with error code {errorcode}")
            sys.exit(errorcode)
    MPI = DummyComm()

import potts_sim
from potts_utils import parse_graph

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
    num_spins, num_edges, edges, opt_cut, opt_energy, mu_max = parse_graph(graph_path)
    return {
        'path': graph_path,
        'name': Path(graph_path).stem,
        'num_spins': num_spins,
        'num_edges': num_edges,
        'edges': edges,
        'opt_cut': opt_cut,
        'opt_energy': opt_energy,
        'mu_max': mu_max,  # Now including mu_max in the graph dict
    }

def safe_eval(expr, env):
    """Safely evaluate an expression with the given environment variables"""
    # This is a simple evaluator - for more complex needs, use numexpr or asteval
    allowed_funcs = {
        'np': np,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float
    }
    
    # Combine the allowed functions with the environment variables
    eval_env = {**allowed_funcs, **env}
    
    # Basic check to prevent harmful code execution
    if any(bad_token in expr for bad_token in ['import', 'exec', 'eval', '__']):
        raise ValueError(f"Potentially unsafe expression: {expr}")
        
    return eval(expr, {"__builtins__": {}}, eval_env)

def parse_range(token):
    """Parse MATLAB-style range notation (start:step:stop) into a list of values"""
    if ':' in token:
        parts = token.split(':')
        if len(parts) == 2:
            # start:stop format
            start, stop = map(float, parts)
            step = 1.0
        elif len(parts) == 3:
            # start:step:stop format
            start, step, stop = map(float, parts)
        else:
            raise ValueError(f"Invalid range format: {token}")
        
        # Calculate number of points (inclusive of start and end)
        n = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(n)]
    else:
        # Single value
        return [float(token)]

def compile_schedule(expr):
    """Return a lambda(num_steps, env) that builds the schedule when called"""
    if expr.startswith("lin(") and expr.endswith(")"):
        start_expr, end_expr = expr[4:-1].split(",", 1)
        return lambda n, env: np.linspace(
            safe_eval(start_expr.strip(), env), 
            safe_eval(end_expr.strip(), env), 
            n
        ).tolist()
    elif expr.startswith("exp(") and expr.endswith(")"):
        parts = expr[4:-1].split(",", 2)
        if len(parts) == 3:
            start_expr, end_expr, factor_expr = [p.strip() for p in parts]
            def exp_schedule(n, env):
                start = safe_eval(start_expr, env)
                end = safe_eval(end_expr, env)
                factor = safe_eval(factor_expr, env)
                base = np.linspace(0, 1, n)
                exp_vals = (np.exp(factor * base) - 1) / (np.exp(factor) - 1)
                return (start + (end - start) * exp_vals).tolist()
            return exp_schedule
        else:
            raise ValueError(f"Exponential schedule needs 3 parameters: {expr}")
    else:
        raise ValueError(f"Unknown schedule format: {expr}")

def get_model_specific_params(config, model_name):
    """Get model-specific simulation parameters, falling back to globals if not specified"""
    global_params = {
        'T': float(config.get('T', 1000.0)),
        'dt': float(config.get('dt', 1e-3)),
        'num_states': int(config.get('num_states', 3)),
        'noise_factor': float(config.get('noise_factor', 1e-4))
    }
    
    # Get model-specific parameters if they exist
    model_params = config.get('models', {}).get(model_name, {})
    
    # Override global params with model-specific ones
    for param in global_params:
        if param in model_params:
            global_params[param] = type(global_params[param])(model_params[param])
    
    return global_params

def expand_param_values(param_values, is_schedule=False):
    """
    Expands parameter values, handling:
    - MATLAB-style ranges for scalars
    - Schedule compilation for schedule strings
    - Expression strings that will be evaluated later (e.g., "-mu_max")
    """
    if not isinstance(param_values, list):
        param_values = [param_values]
    
    if is_schedule:
        # For schedules: compile each string to a lambda
        return [compile_schedule(str(val)) for val in param_values]
    else:
        # For scalars: expand MATLAB-style ranges
        expanded = []
        for val in param_values:
            if isinstance(val, str):
                if ':' in val:
                    expanded.extend(parse_range(val))
                elif any(token in val for token in ['mu_max', 'alpha']):
                    # This is an expression to be evaluated later with mu_max or alpha
                    expanded.append(val)  # Keep as string
                else:
                    # Try to convert to float if it's a simple number string
                    try:
                        expanded.append(float(val))
                    except ValueError:
                        # If conversion fails, keep it as a string for later evaluation
                        expanded.append(val)
            else:
                expanded.append(float(val))
        return expanded

def generate_param_sets(model_type, model_params, T, dt):
    """Generate all parameter combinations for a model type"""
    num_steps = int(np.floor(T / dt))
    
    # Common function to process schedule definitions including linked schedules
    def process_schedule_param(param_name, default_value):
        # Check if this is a linked schedule
        if isinstance(model_params.get(param_name), dict):
            schedule_def = model_params[param_name]
            # This schedule is based on another with a factor
            if 'based_on' in schedule_def and 'factor' in schedule_def:
                base_schedule = schedule_def['based_on']
                factors = expand_param_values(schedule_def['factor'])
                return {
                    'type': 'linked',
                    'base_schedule': base_schedule,
                    'factors': factors
                }
            else:
                raise ValueError(f"Invalid linked schedule definition for {param_name}")
        else:
            # Store schedule expressions as strings, not compiled lambdas
            schedules = model_params.get(param_name, default_value)
            if not isinstance(schedules, list):
                schedules = [schedules]
            return {
                'type': 'direct',
                'schedules': schedules  # Just the raw strings
            }
    
    # For each model type, define how to generate parameter sets
    if model_type == ModelType.POLYNOMIAL:
        poly_orders = [int(po) for po in expand_param_values(model_params.get('poly_order', [3]))]
        
        # Process schedules
        beta_info = process_schedule_param('beta_schedule', ["lin(0,1)"])
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        
        # Handle the case where both are directly specified
        if beta_info['type'] == 'direct' and gamma_info['type'] == 'direct':
            for poly_order, beta_expr, gamma_expr in itertools.product(
                    poly_orders, beta_info['schedules'], gamma_info['schedules']):
                param_id = f"po{poly_order}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,  # Store expression, not function
                    'gamma_expr': gamma_expr
                })
        
        # Handle gamma based on beta with factor
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linked' and gamma_info['base_schedule'] == 'beta_schedule':
            for poly_order, beta_expr, factor in itertools.product(
                    poly_orders, beta_info['schedules'], gamma_info['factors']):
                param_id = f"po{poly_order}_b{beta_expr}_gf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_factor': factor,
                    'gamma_based_on': 'beta'
                })
        
        # Handle beta based on gamma with factor
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'linked' and beta_info['base_schedule'] == 'gamma_schedule':
            for poly_order, gamma_expr, factor in itertools.product(
                    poly_orders, gamma_info['schedules'], beta_info['factors']):
                param_id = f"po{poly_order}_g{gamma_expr}_bf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_factor': factor,
                    'beta_based_on': 'gamma'
                })
        
        else:
            raise ValueError("Invalid schedule linkage configuration")
            
        return param_sets
        
    elif model_type == ModelType.NEC:
        # Expand all parameter ranges
        alpha_rates = expand_param_values(model_params.get('alpha_rate', [1e-2]))
        gammas = expand_param_values(model_params.get('gamma', [1.0]))
        r_targets = expand_param_values(model_params.get('r_target', [2.0]))
        initial_alphas = expand_param_values(model_params.get('initial_alpha', [1.0]))
        
        param_sets = []
        for alpha_rate, gamma, r_target, initial_alpha in itertools.product(
                alpha_rates, gammas, r_targets, initial_alphas):
            # Create a unique identifier for this parameter set
            # Handle the case where initial_alpha is an expression string
            ia_str = initial_alpha if isinstance(initial_alpha, str) else f"{initial_alpha}"
            param_id = f"ar{alpha_rate:.2e}_g{gamma}_rt{r_target}_ia{ia_str}"
            
            param_sets.append({
                'id': param_id,
                'alpha_rate': alpha_rate,
                'gamma': gamma,
                'r_target': r_target,
                'initial_alpha': initial_alpha,
                'initial_alpha_expr': initial_alpha if isinstance(initial_alpha, str) else None
            })
        return param_sets
        
    elif model_type == ModelType.SIGMOID:
        # Expand scalar parameters
        alphas = expand_param_values(model_params.get('alpha', [-1.0]))
        
        # Process schedules
        beta_info = process_schedule_param('beta_schedule', ["lin(0,1)"])
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        
        # Handle the case where both are directly specified
        if beta_info['type'] == 'direct' and gamma_info['type'] == 'direct':
            for alpha, beta_expr, gamma_expr in itertools.product(
                    alphas, beta_info['schedules'], gamma_info['schedules']):
                param_id = f"a{alpha}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })
        
        # Handle gamma based on beta with factor
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linked' and gamma_info['base_schedule'] == 'beta_schedule':
            for alpha, beta_expr, factor in itertools.product(
                    alphas, beta_info['schedules'], gamma_info['factors']):
                param_id = f"a{alpha}_b{beta_expr}_gf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_factor': factor,
                    'gamma_based_on': 'beta'
                })
        
        # Handle beta based on gamma with factor
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'linked' and beta_info['base_schedule'] == 'gamma_schedule':
            for alpha, gamma_expr, factor in itertools.product(
                    alphas, gamma_info['schedules'], beta_info['factors']):
                param_id = f"a{alpha}_g{gamma_expr}_bf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_factor': factor,
                    'beta_based_on': 'gamma'
                })
        
        else:
            raise ValueError("Invalid schedule linkage configuration")
            
        return param_sets
        
    elif model_type == ModelType.FIXED_AMPLITUDE:
        # Process gamma schedule
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,2)"])
        
        param_sets = []
        if gamma_info['type'] == 'direct':
            for gamma_expr in gamma_info['schedules']:
                param_id = f"g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'gamma_expr': gamma_expr
                })
        else:
            raise ValueError("FIXED_AMPLITUDE model only supports direct gamma_schedule specification")
            
        return param_sets
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_task(graph, model_type, param_set, seed, T, dt, num_states, noise_factor):
    """Run a single task with the given parameters"""
    num_spins = graph['num_spins']
    edges = graph['edges']
    mu_max = graph['mu_max']
    start_time = time.time()
    
    # Calculate num_steps
    num_steps = int(np.floor(T / dt))
    
    # Generate a deterministic seed that's unique for this task
    task_seed = int(hashlib.md5(f"{graph['path']}_{model_type.name}_{param_set['id']}_{seed}".encode()).hexdigest(), 16) % (2**31)
    
    # Prepare environment for schedule evaluation
    env = {
        'mu_max': mu_max,
        'np': np
    }
    
    # Add model-specific parameters to environment
    for key, value in param_set.items():
        if key not in ['id', 'beta_fn', 'gamma_fn', 'beta_factor', 'gamma_factor', 'beta_based_on', 'gamma_based_on']:
            env[key] = value
    
    # Evaluate schedules based on model type requirements
    if model_type == ModelType.POLYNOMIAL:
        # For POLYNOMIAL model
        if 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            # gamma = factor * beta
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * b for b in beta_schedule]
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            # beta = factor * gamma
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * g for g in gamma_schedule]
        else:
            # Both schedules directly defined
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
        
        res = potts_sim.run_polynomial(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['poly_order'],
            beta_schedule, gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    elif model_type == ModelType.NEC:
        # For NEC model
        # Handle special case for initial_alpha if it's an expression
        if 'initial_alpha_expr' in param_set and param_set['initial_alpha_expr']:
            # Evaluate the expression (e.g., "mu_max * -1")
            initial_alpha = safe_eval(param_set['initial_alpha_expr'], env)
        else:
            initial_alpha = param_set['initial_alpha']
            
        initial_alpha_arr = [initial_alpha] * num_spins
        
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
        if 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            # gamma = factor * beta
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * b for b in beta_schedule]
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            # beta = factor * gamma
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * g for g in gamma_schedule]
        else:
            # Both schedules directly defined
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            
        res = potts_sim.run_sigmoid(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            param_set['alpha'],
            beta_schedule, gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True
        )
    
    elif model_type == ModelType.FIXED_AMPLITUDE:
        # For FIXED_AMPLITUDE model
        gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
        
        res = potts_sim.run_fixed_amplitude(
            T, dt, num_spins, num_states,
            edges,
            noise_factor, task_seed,
            gamma_schedule,
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
        "num_spins": num_spins,
        "model": model_type.name,
        "param_id": param_set['id'],
        "seed": seed,
        "cut_value": best_cut,
        "energy": best_energy,
        "step": best_step,
        "opt_cut": graph['opt_cut'],
        "opt_energy": graph['opt_energy'],
        "cut_gap": best_cut - graph['opt_cut'] if graph['opt_cut'] is not None else None,
        "energy_gap": best_energy - graph['opt_energy'] if graph['opt_energy'] is not None else None,
        "runtime": elapsed_time,
        "T": T,
        "dt": dt,
        "num_steps": num_steps,
        "mu_max": mu_max,
    }
    
    # Add model-specific parameters to the result
    if model_type == ModelType.POLYNOMIAL:
        result.update({
            "poly_order": param_set['poly_order']
        })
        # Add schedule information based on what was used
        if 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_factor": param_set['gamma_factor']
            })
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            result.update({
                "gamma_schedule": param_set['gamma_expr'],
                "beta_factor": param_set['beta_factor']
            })
        else:
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_schedule": param_set['gamma_expr']
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
            "alpha": param_set['alpha']
        })
        # Add schedule information based on what was used
        if 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_factor": param_set['gamma_factor']
            })
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            result.update({
                "gamma_schedule": param_set['gamma_expr'],
                "beta_factor": param_set['beta_factor']
            })
        else:
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_schedule": param_set['gamma_expr']
            })
    elif model_type == ModelType.FIXED_AMPLITUDE:
        result.update({
            "gamma_schedule": param_set['gamma_expr']
        })
    
    return result

def main():
    # Initialize MPI or dummy
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = MPI
        rank = 0
        size = 1

    # Only rank 0 parses command line arguments
    if rank == 0:
        parser = argparse.ArgumentParser(description='Run Potts model parameter sweep')
        parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
        # Change how we store the argument to track when the flag is used
        parser.add_argument('--estimate_wall_time', nargs='?', const=True, type=int, metavar='RANKS',
                           help='Only estimate wall time without running the sweep; optionally specify number of ranks')
        args = parser.parse_args()
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract config basename for filenames
        config_basename = os.path.splitext(os.path.basename(args.config))[0]
        
        # Extract general parameters
        num_runs = int(config.get('num_runs', 100))
        graph_path_spec = config.get('graph_path', 'graphs/')
        out_dir = config.get('out_dir', 'results/')
        
        # Make sure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Identify graph files, handling both string and list specifications
        graph_files = []
        
        # Convert single path to list for uniform handling
        if isinstance(graph_path_spec, str):
            graph_path_spec = [graph_path_spec]
            
        for path in graph_path_spec:
            if os.path.isdir(path):
                # If it's a directory, add all .col files
                graph_files.extend(glob.glob(os.path.join(path, "*.col")))
            elif os.path.isfile(path):
                # If it's a file, add it directly (if it ends with .col)
                if path.endswith('.col'):
                    graph_files.append(path)
            else:
                print(f"Warning: Path {path} is neither a directory nor a file, skipping")
        
        # Remove duplicates while preserving order
        graph_files = list(dict.fromkeys(graph_files))
        
        if not graph_files:
            print(f"Error: No .col files found in the specified paths")
            if use_mpi:
                comm.Abort(1)
            else:
                sys.exit(1)
            
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
                
                # Get model-specific simulation parameters
                sim_params = get_model_specific_params(config, model_name)
                
                # Generate all parameter sets for this model
                param_sets = generate_param_sets(model_type, model_params, sim_params['T'], sim_params['dt'])
                
                # For each parameter set, add tasks for all seeds
                for param_set in param_sets:
                    for seed in range(num_runs):
                        tasks.append({
                            'graph': graph,
                            'model_type': model_type,
                            'param_set': param_set,
                            'seed': seed,
                            'T': sim_params['T'],
                            'dt': sim_params['dt'],
                            'num_states': sim_params['num_states'],
                            'noise_factor': sim_params['noise_factor']
                        })
        
        print(f"Generated {len(tasks)} tasks")
        
        # Estimate total wall time
        total_steps = sum(int(np.floor(task['T'] / task['dt'])) for task in tasks)
        
        # Determine number of ranks to use for estimation
        if args.estimate_wall_time is not None:
            if isinstance(args.estimate_wall_time, bool):
                # No value provided, use current MPI size
                estimation_ranks = size
            else:
                # User specified a number of ranks
                estimation_ranks = args.estimate_wall_time
        else:
            # Flag not used at all
            estimation_ranks = size
        
        steps_per_rank = int(total_steps / max(1, estimation_ranks))
        
        expected_runtime_us = steps_per_rank * ASSUMED_TIME_PER_STEP_US
        expected_runtime_sec = expected_runtime_us / 1e6
        expected_runtime = timedelta(seconds=int(expected_runtime_sec))
        
        # Print wall time estimate in a more readable format
        print("\n" + "="*80)
        print(f"WALL TIME ESTIMATE")
        print("-"*80)
        print(f"Tasks                : {len(tasks):,}")
        print(f"Total steps          : {total_steps:,}")
        print(f"Ranks for estimate   : {estimation_ranks}")
        print(f"Steps per rank       : {steps_per_rank:,}")
        print(f"Time per step        : {ASSUMED_TIME_PER_STEP_US} µs (assumed)")
        print(f"Expected runtime     : {expected_runtime}")
        print("="*80 + "\n")
        
        # If only estimating wall time, exit here
        if args.estimate_wall_time is not None:  # This now covers both True and integer values
            print("Walltime estimation completed. Exiting without running sweep.")
            if use_mpi:
                # Signal all ranks to exit
                tasks = []  # Empty task list signals exit
                out_dir = None
            else:
                sys.exit(0)
        
        # Save run configuration
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'git_revision': get_git_revision(),
            'config_file': args.config,
            'num_tasks': len(tasks),
            'num_graphs': len(graph_files),
            'num_ranks': size,
            'estimated_runtime_seconds': expected_runtime_sec,
            'assumed_time_per_step_us': ASSUMED_TIME_PER_STEP_US
        }
        
        run_info_file = os.path.join(out_dir, f"run_info_{config_basename}.yaml")
        with open(run_info_file, 'w') as f:
            yaml.dump(run_info, f)
    else:
        # Non-root ranks initialize these variables
        tasks = None
        out_dir = None
        run_info = None
        args = None
        config_basename = None
    
    # Broadcast task list, output directory and config basename to all ranks
    tasks = comm.bcast(tasks, root=0)
    out_dir = comm.bcast(out_dir, root=0)
    config_basename = comm.bcast(config_basename, root=0)
    
    # If tasks is empty and we're using MPI, it means we're just estimating wall time
    if use_mpi and not tasks:
        return  # Exit the function
    
    # Record the actual start time of computation
    computation_start_time = time.time()
    
    # Divide tasks among ranks using simple strided allocation
    my_tasks = tasks[rank::size]
    print(f"Rank {rank}: Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    local_results = []
    processing_start_time = time.time()
    last_print_time = processing_start_time
    last_print_task = 0
    total_steps_processed = 0
    
    for i, task in enumerate(my_tasks):
        try:
            result = run_task(**task)
            local_results.append(result)
            
            # Track steps for performance metrics
            task_steps = int(np.floor(task['T'] / task['dt']))
            total_steps_processed += task_steps
            
            if i % 10 == 0:
                now = time.time()
                elapsed = now - processing_start_time
                
                # Calculate time per step metrics (in microseconds)
                overall_time_per_step_us = (elapsed * 1e6) / total_steps_processed if total_steps_processed > 0 else 0
                
                # Calculate metrics since last print
                if i > last_print_task:
                    time_since_last = now - last_print_time
                    tasks_since_last = i - last_print_task
                    steps_since_last = sum(int(np.floor(my_tasks[j]['T'] / my_tasks[j]['dt'])) 
                                          for j in range(last_print_task, i))
                    recent_time_per_step_us = (time_since_last * 1e6) / steps_since_last if steps_since_last > 0 else 0
                else:
                    recent_time_per_step_us = 0
                
                # Estimate remaining time (using the recent time per step)
                remaining_tasks = len(my_tasks) - (i + 1)
                remaining_steps = sum(int(np.floor(my_tasks[j]['T'] / my_tasks[j]['dt'])) 
                                     for j in range(i + 1, len(my_tasks)))
                eta_seconds = (remaining_steps * recent_time_per_step_us / 1e6) if recent_time_per_step_us > 0 else 0
                eta = timedelta(seconds=int(eta_seconds))
                
                # Update for next iteration
                last_print_time = now
                last_print_task = i
                
                print(f"Rank {rank}: Completed {i+1}/{len(my_tasks)} tasks | "
                      f"Runtime: {timedelta(seconds=int(elapsed))} | "
                      f"ETA: {eta} | "
                      f"Time/step: {overall_time_per_step_us:.1f} µs avg, {recent_time_per_step_us:.1f} µs current")
        except Exception as e:
            print(f"Rank {rank}: Error processing task {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate final performance metrics for this rank
    computation_end_time = time.time()
    local_computation_time = computation_end_time - computation_start_time
    #local_time_per_step_us = (local_computation_time * 1e6) / total_steps_processed if total_steps_processed > 0 else 0
    
    # Convert results to DataFrame
    if local_results:
        local_df = pd.DataFrame(local_results)
    else:
        # Create empty DataFrame with expected columns if no results
        local_df = pd.DataFrame()
    
    # Store local results to disk (useful for recovery if job fails)
    local_out = os.path.join(out_dir, f"results_rank{rank}_{config_basename}.parquet")
    if not local_df.empty:
        local_df.to_parquet(local_out, index=False)
    
    # Gather all DataFrames and performance metrics at rank 0
    all_dfs = comm.gather(local_df, root=0)
    all_steps_processed = comm.gather(total_steps_processed, root=0)
    all_computation_times = comm.gather(local_computation_time, root=0)
    
    # Rank 0 combines and saves the final results
    if rank == 0:
        # Record the final end time
        final_end_time = time.time()
        total_walltime = final_end_time - computation_start_time
        
        # Calculate overall average time per step
        total_steps = sum(all_steps_processed)
        avg_time_per_step_us = 0
        if total_steps > 0:
            # Calculate average across all ranks
            avg_time_per_step_us = (sum(all_computation_times) * 1e6) / total_steps
        
        # Filter out empty DataFrames and concatenate
        all_dfs = [df for df in all_dfs if not df.empty]
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save the combined results
            combined_out = os.path.join(out_dir, f"results_{config_basename}.parquet")
            combined_df.to_parquet(combined_out, index=False)
            
            print(f"Saved combined results with {len(combined_df)} rows to {combined_out}")
            
            # Clean up individual rank result files
            rank_files = glob.glob(os.path.join(out_dir, f"results_rank*_{config_basename}.parquet"))
            for file in rank_files:
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Warning: Could not remove {file}: {e}")
            print(f"Cleaned up {len(rank_files)} individual rank result files")
        else:
            print("Warning: No results were collected from any rank")
        
        # Update run_info with actual performance metrics
        run_info = {
            'timestamp': datetime.now().isoformat(),
            'git_revision': get_git_revision(),
            'config_file': args.config,
            'num_tasks': len(tasks),
            'num_graphs': len(graph_files),
            'num_ranks': size,
            'total_steps_processed': total_steps,
            'assumed_time_per_step_us': ASSUMED_TIME_PER_STEP_US,
            'estimated_runtime_seconds': expected_runtime_sec,
            'actual_time_per_step_us': avg_time_per_step_us,
            'actual_walltime_seconds': total_walltime
        }
        
        # Save the updated run_info with actual metrics
        run_info_file = os.path.join(out_dir, f"run_info_{config_basename}.yaml")
        with open(run_info_file, 'w') as f:
            yaml.dump(run_info, f, sort_keys=False)
        
        print(f"Finished sweep in {timedelta(seconds=int(total_walltime))}")
        print(f"Average time per step: {avg_time_per_step_us:.2f} µs")

if __name__ == "__main__":
    main()
