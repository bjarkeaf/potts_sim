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

"""
Flags:
- '--config': Path to the configuration YAML file.
- `--estimate_wall_time`: Estimate wall time without running the sweep.
    - If provided with a value, it specifies the number of ranks to use for estimation.
    - If provided without a value, it uses the current MPI size.
- `--plot_schedules`: Generate visualizations of the schedules that would be used in simulations.
    - Saves plots to 'schedule_plots_{config name}/' directory.
"""

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
from cim_sim import run_cim_from_graph

class ModelType(Enum):
    """Enum for the different Potts model types"""
    QPDC = "q-pdc" # alias for polynomial
    NEC = "nec"
    POLYNOMIAL = "polynomial" 
    SIGMOID = "sigmoid"
    FIXED_AMPLITUDE = "fixed_amplitude"
    CIM = "cim"  # Add CIM model type

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
    num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max = parse_graph(graph_path)
    return {
        'path': graph_path,
        'name': Path(graph_path).stem,
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'edges': edges,
        'opt_cut_dict': opt_cut_dict,
        'opt_energy_dict': opt_energy_dict,
        'mu_max': mu_max,
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
    """Parse MATLAB-style range notation (start:step:stop) into a list of values.
    Supports expressions using constants like gamma_th, e.g. '0:0.04:2*gamma_th'"""
    constants_env = {
        'gamma_th': (256/27)**(1/4),
        'np': np
    }

    def eval_part(part):
        part = part.strip()
        try:
            return float(part)
        except ValueError:
            return safe_eval(part, constants_env)

    if ':' in token:
        parts = token.split(':')
        if len(parts) == 2:
            # start:stop format
            start, stop = eval_part(parts[0]), eval_part(parts[1])
            step = 1.0
        elif len(parts) == 3:
            # start:step:stop format
            start, step, stop = eval_part(parts[0]), eval_part(parts[1]), eval_part(parts[2])
        else:
            raise ValueError(f"Invalid range format: {token}")

        # Calculate number of points (inclusive of start and end)
        n = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(n)]
    else:
        # Single value
        return [eval_part(token)]

def compile_schedule(expr):
    """Return a lambda(num_steps, env) that builds the schedule when called"""
    if expr.startswith("lin(") and expr.endswith(")"):
        start_expr, end_expr = expr[4:-1].split(",", 1)
        return lambda n, env: np.linspace(
            safe_eval(start_expr.strip(), env),
            safe_eval(end_expr.strip(), env),
            n
        ).tolist()
    elif expr.startswith("linspan(") and expr.endswith(")"):
        # Parse arguments: linspan(start_expr, span_expr)
        args_str = expr[8:-1]

        # Handle nested function calls by finding comma at depth 0
        depth = 0
        split_pos = -1
        for i, char in enumerate(args_str):
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 0:
                split_pos = i
                break

        if split_pos == -1:
            raise ValueError(f"linspan schedule needs exactly 2 parameters: {expr}")

        start_expr = args_str[:split_pos].strip()
        span_expr = args_str[split_pos+1:].strip()

        def linspan_schedule(n, env):
            start = safe_eval(start_expr, env)
            span = safe_eval(span_expr, env)
            end = start + span
            return np.linspace(start, end, n).tolist()

        return linspan_schedule
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
    elif expr.startswith("const(") and expr.endswith(")"):
        value_expr = expr[6:-1].strip()
        return lambda n, env: [safe_eval(value_expr, env)] * n
    else:
        raise ValueError(f"Unknown schedule format: {expr}")

def get_model_specific_params(config, model_name):
    """Get model-specific simulation parameters, falling back to globals if not specified

    Returns raw values (can be lists/strings for sweeps). Caller is responsible for expansion.
    """
    global_params = {
        'T': config.get('T', 1000.0),
        'dt': config.get('dt', 1e-3),
        'num_states': config.get('num_states', 3),
        'noise_factor': config.get('noise_factor', 1e-4)
    }

    # Get model-specific parameters if they exist
    model_params = config.get('models', {}).get(model_name, {})

    # Handle case where model_params is None (YAML interprets "NEC:" with nothing as None)
    if model_params is None:
        model_params = {}

    # Override global params with model-specific ones
    for param in global_params:
        if param in model_params:
            global_params[param] = model_params[param]

    return global_params

def load_hyperparams_from_csv(csv_path):
    """
    Load best hyperparameters from a CSV file generated by save_best_hyperparams_csv().

    Parameters:
    - csv_path: Path to the CSV file

    Returns:
    - Dictionary mapping (model, graph) tuples to hyperparameter dictionaries
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Hyperparam table file not found: {csv_path}")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['Model', 'Graph', 'param_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")

    # Build the hyperparam table
    hyperparam_table = {}

    # Mapping from pretty-print names (used in CSV/results) to backend enum names
    # Also includes backend names for robustness (CSV may use either format)
    pretty_to_backend = {
        # Pretty names
        'Polynomial PM': 'POLYNOMIAL',
        'q-PDC': 'QPDC',
        'q-SHIL': 'FIXED_AMPLITUDE',
        'Sigmoid PM': 'SIGMOID',
        'Sigmoid IM': 'CIM',
        # Backend names (identity mapping)
        'NEC': 'NEC',
        'POLYNOMIAL': 'POLYNOMIAL',
        'QPDC': 'QPDC',
        'FIXED_AMPLITUDE': 'FIXED_AMPLITUDE',
        'SIGMOID': 'SIGMOID',
        'CIM': 'CIM',
    }

    for _, row in df.iterrows():
        model_pretty = row['Model']
        graph = row['Graph']
        param_id = row['param_id']

        # Convert pretty-print name to backend name
        model_backend = pretty_to_backend.get(model_pretty, model_pretty)

        # Map model name to ModelType enum
        try:
            model_type = ModelType[model_backend]
        except KeyError:
            print(f"Warning: Unknown model type '{model_pretty}' (backend: '{model_backend}') in CSV, skipping")
            continue

        # Parse the param_id to extract hyperparameters
        params = parse_param_id(param_id, model_type)

        # Store in the table using backend name (matches YAML config model names)
        hyperparam_table[(model_backend, graph)] = params

    print(f"Loaded hyperparameters for {len(hyperparam_table)} model-graph combinations from {csv_path}")

    return hyperparam_table

def print_hyperparam_summary(config, graph_files, hyperparam_table):
    """
    Print a detailed summary of which hyperparameters would be loaded from the hyperparam table.

    Parameters:
    - config: Configuration dictionary from YAML
    - graph_files: List of graph file paths
    - hyperparam_table: Dictionary from load_hyperparams_from_csv() or None
    """
    print("\n" + "="*80)
    print("DRY-RUN: Parameter preview for configuration")
    print("="*80)

    if hyperparam_table:
        # Show hyperparam table info
        hyperparam_table_path = config.get('hyperparam_table', 'Unknown')
        print(f"Hyperparameter table: {hyperparam_table_path}")
        print(f"Loaded: {len(hyperparam_table)} model-graph combinations\n")

        # Show global parameters from YAML
        print("Global Parameters (from YAML, override hyperparam table):")
        print(f"  T: {config.get('T', 'default')}")
        print(f"  dt: {config.get('dt', 'default')}")
        print(f"  num_runs: {config.get('num_runs', 100)}")
        print(f"  num_states: {config.get('num_states', 3)}")
        print(f"  noise_factor: {config.get('noise_factor', 1e-4)}")
        print()

        # Track coverage
        total_combinations = 0
        found_combinations = 0
        total_param_sets = 0

        # Process each graph
        for graph_file in graph_files:
            graph_name = Path(graph_file).stem
            print(f"Graph: {graph_name}")
            print("-"*80)

            # Process each model
            for model_name, model_params in config.get('models', {}).items():
                total_combinations += 1

                # Handle case where model_params is None (YAML interprets "MODEL:" with nothing as None)
                if model_params is None:
                    model_params = {}

                try:
                    model_type = ModelType[model_name]
                except KeyError:
                    print(f"  ✗ {model_name} - UNKNOWN MODEL TYPE")
                    print()
                    continue

                # Get simulation parameters
                sim_params = get_model_specific_params(config, model_name)

                # Expand T and dt to support sweeps
                T_raw = sim_params['T']
                if isinstance(T_raw, list):
                    T_values = expand_param_values(T_raw)
                elif isinstance(T_raw, str):
                    T_values = expand_param_values([T_raw])
                else:
                    T_values = [float(T_raw)]

                dt_raw = sim_params['dt']
                if isinstance(dt_raw, list):
                    dt_values = expand_param_values(dt_raw)
                elif isinstance(dt_raw, str):
                    dt_values = expand_param_values([dt_raw])
                else:
                    dt_values = [float(dt_raw)]

                # Get hyperparams from table if available
                hyperparams_from_csv = None
                if (model_name, graph_name) in hyperparam_table:
                    found_combinations += 1
                    hyperparams_from_csv = hyperparam_table[(model_name, graph_name)]

                # Generate actual param_sets (with merging) for all (T, dt) combinations
                all_param_sets = []
                try:
                    for T in T_values:
                        for dt in dt_values:
                            param_sets = generate_param_sets(model_type, model_params, T, dt, hyperparams_from_csv)
                            # Add T, dt, num_steps to each param_set
                            for param_set in param_sets:
                                param_set['T'] = T
                                param_set['dt'] = dt
                                param_set['num_steps'] = int(np.floor(T / dt))
                                all_param_sets.append(param_set)
                    param_sets = all_param_sets
                except Exception as e:
                    print(f"  ✗ {model_name} - ERROR: {e}")
                    print()
                    continue

                total_param_sets += len(param_sets)

                if hyperparams_from_csv:
                    print(f"  ✓ {model_name} (Hyperparam table + Config merge)")
                    print(f"    Hyperparam table param_id: {hyperparams_from_csv['id']}")
                else:
                    print(f"  ○ {model_name} (Config only - not in hyperparam table)")

                print(f"    Generated {len(param_sets)} parameter set(s):")

                # Show each param_set
                for i, param_set in enumerate(param_sets):
                    if len(param_sets) > 1:
                        print(f"\n    Set {i+1}/{len(param_sets)}:")
                        print(f"      param_id: {param_set.get('id', 'N/A')}")

                    # Determine source for each parameter
                    print(f"      parameters:")

                    # Show global/simulation parameters first (T, dt, num_steps, num_states, noise_factor)
                    for key in ['T', 'dt', 'num_steps', 'num_states', 'noise_factor']:
                        if key in param_set:
                            value = param_set[key]
                            if isinstance(value, float):
                                value_str = f"{value:.4g}"
                            else:
                                value_str = str(value)

                            # Determine if this came from model-specific or global config
                            # num_steps is always computed, others can be model or global
                            if key == 'num_steps':
                                source = "GLOBAL CONFIG"  # Derived from T/dt
                            elif key in model_params and model_params[key] is not None:
                                source = "MODEL CONFIG"
                            else:
                                source = "GLOBAL CONFIG"

                            print(f"        {key}={value_str} [{source}]")

                    # Key hyperparameters to show
                    param_keys = ['poly_order', 'alpha', 'alpha_rate', 'r_target', 'initial_alpha',
                                  'B_num_vertices', 'B', 'zeta', 'beta_expr', 'gamma_expr',
                                  'gamma_factor', 'beta_factor']

                    for key in param_keys:
                        if key in param_set:
                            value = param_set[key]

                            # Determine source - check if parameter was explicitly set
                            # For _expr parameters, check the corresponding _schedule parameter
                            config_key = key
                            if key.endswith('_expr'):
                                # gamma_expr comes from gamma_schedule, beta_expr from beta_schedule
                                config_key = key.replace('_expr', '_schedule')

                            # Check if explicitly in model config
                            in_model_config = config_key in model_params and model_params[config_key] is not None

                            # Check if in hyperparam table
                            in_csv = hyperparams_from_csv and key in hyperparams_from_csv and hyperparams_from_csv[key] == value

                            # Determine source
                            if in_model_config and in_csv:
                                source = "MODEL CONFIG (same as table)"
                            elif in_model_config:
                                source = "MODEL CONFIG"
                            elif in_csv:
                                source = "HYPERPARAM TABLE"
                            else:
                                source = "DEFAULT"

                            # Format the value
                            if isinstance(value, float):
                                value_str = f"{value:.4g}"
                            else:
                                value_str = str(value)

                            # Add special annotations
                            if key == 'gamma_factor' and param_set.get('gamma_is_prototype'):
                                value_str += " (prototype)"
                            if key == 'beta_factor' and param_set.get('beta_is_prototype'):
                                value_str += " (prototype)"

                            print(f"        {key}={value_str} [{source}]")

                print()

        # Summary
        print("Summary:")
        num_runs = config.get('num_runs', 100)
        total_tasks = total_param_sets * num_runs

        print(f"  Total parameter sets: {total_param_sets}")
        print(f"  Total tasks: {total_tasks} ({total_param_sets} param_sets × {num_runs} runs)")
        coverage_pct = (found_combinations / total_combinations * 100) if total_combinations > 0 else 0
        print(f"  Hyperparam table coverage: {found_combinations}/{total_combinations} ({coverage_pct:.1f}%)")
        print()

        if found_combinations == total_combinations:
            print("✓ All model-graph combinations found in hyperparam table")
        else:
            print(f"⚠ Warning: {total_combinations - found_combinations} model-graph combinations missing from hyperparam table")
            print("  Missing combinations will use sweep mode from YAML config")

    else:
        # No hyperparam table - show what sweep would be generated
        print("No hyperparam_table specified in config")
        print()
        print("Sweep Mode Preview:")
        print("  Will generate parameter sweeps based on YAML model configurations")
        print()

        for model_name, model_params in config.get('models', {}).items():
            try:
                model_type = ModelType[model_name]
            except KeyError:
                continue

            print(f"  {model_name}:")

            # Show key parameters that will be swept
            if model_type == ModelType.POLYNOMIAL or model_type == ModelType.QPDC:
                poly_orders = model_params.get('poly_order', [3])
                beta_schedules = model_params.get('beta_schedule', ["lin(0,1)"])
                gamma_config = model_params.get('gamma_schedule', ["lin(0,1)"])
                print(f"    poly_order: {poly_orders}")
                print(f"    beta_schedule: {beta_schedules}")
                print(f"    gamma_schedule: {gamma_config}")

            elif model_type == ModelType.NEC:
                print(f"    alpha_rate: {model_params.get('alpha_rate', [1e-2])}")
                print(f"    r_target: {model_params.get('r_target', [2.0])}")
                print(f"    gamma_schedule: {model_params.get('gamma_schedule', ['lin(0,1)'])}")

            elif model_type == ModelType.SIGMOID:
                print(f"    alpha: {model_params.get('alpha', [-1.0])}")
                print(f"    beta_schedule: {model_params.get('beta_schedule', ['lin(0,1)'])}")
                print(f"    gamma_schedule: {model_params.get('gamma_schedule', ['lin(0,1)'])}")

            elif model_type == ModelType.CIM:
                print(f"    alpha: {model_params.get('alpha', [-10.0])}")
                if 'B_num_vertices' in model_params:
                    print(f"    B_num_vertices: {model_params.get('B_num_vertices')}")
                else:
                    print(f"    B: {model_params.get('B', [0.18])}")
                print(f"    zeta: {model_params.get('zeta', [0.6])}")

            print()

    print("="*80 + "\n")

def parse_param_id(param_id, model_type):
    """
    Parse a param_id string to extract hyperparameter values.

    Parameters:
    - param_id: String identifier like "po3_blin(0,1)_gf0.8" or "ar1.00e-02_rt2.1_ia-mu_max_gpf8.0"
    - model_type: ModelType enum

    Returns:
    - Dictionary with extracted hyperparameter values
    """
    params = {}

    # Split by underscore, looking ahead for parameter prefixes
    # Match common prefixes: po, ar, rt, ia, gf, gpf, bf, bpf, a, Bnv, B, z
    # Also match schedule patterns: blin(, glin(, bconst(, gconst(, bexp(, gexp(
    parts = re.split(r'_(?=(?:po|ar|rt|ia|gf|gpf|bf|bpf|Bnv|B(?!nv)|z|a(?!r)|[bg](?:lin|const|exp)\())', param_id)

    for part in parts:
        # Polynomial order: po3
        if part.startswith('po'):
            try:
                params['poly_order'] = int(float(part[2:]))
            except ValueError:
                params['poly_order'] = part[2:]  # Keep as string if it's an expression

        # Beta schedule: blin(...) or bconst(...) or bexp(...) or blinspan(...)
        elif part.startswith('b') and ('lin(' in part or 'const(' in part or 'exp(' in part or 'linspan(' in part):
            # Extract the full schedule expression
            params['beta_expr'] = part[1:]

        # Gamma schedule: glin(...) or gconst(...) or gexp(...) or glinspan(...)
        elif part.startswith('g') and ('lin(' in part or 'const(' in part or 'exp(' in part or 'linspan(' in part):
            params['gamma_expr'] = part[1:]

        # Gamma factor: gf0.8 or gpf8.0 (prototype factor)
        elif part.startswith('gpf'):
            params['gamma_factor'] = float(part[3:])
            params['gamma_is_prototype'] = True
        elif part.startswith('gf'):
            params['gamma_factor'] = float(part[2:])
            params['gamma_based_on'] = 'beta'

        # Beta factor: bf0.5 or bpf5.0 (prototype factor)
        elif part.startswith('bpf'):
            params['beta_factor'] = float(part[3:])
            params['beta_is_prototype'] = True
        elif part.startswith('bf'):
            params['beta_factor'] = float(part[2:])
            params['beta_based_on'] = 'gamma'

        # Alpha rate: ar1.00e-02
        elif part.startswith('ar'):
            params['alpha_rate'] = float(part[2:])

        # R target: rt2.1
        elif part.startswith('rt'):
            params['r_target'] = float(part[2:])

        # Initial alpha: ia-mu_max or ia1.0
        elif part.startswith('ia'):
            alpha_val = part[2:]
            if alpha_val == '-mu_max' or 'mu_max' in alpha_val:
                params['initial_alpha_expr'] = alpha_val
                params['initial_alpha'] = alpha_val  # Will be evaluated later
            else:
                params['initial_alpha'] = float(alpha_val)

        # Alpha (for SIGMOID/CIM): a-50.0
        elif part.startswith('a') and not part.startswith('ar'):
            try:
                params['alpha'] = float(part[1:])
            except ValueError:
                pass

        # B_num_vertices: Bnv175.0
        elif part.startswith('Bnv'):
            params['B_num_vertices'] = float(part[3:])

        # B: B0.18
        elif part.startswith('B') and not part.startswith('Bnv'):
            params['B'] = float(part[1:])

        # Zeta: z0.6
        elif part.startswith('z'):
            try:
                params['zeta'] = float(part[1:])
            except ValueError:
                pass

    # Add the original param_id for reference
    params['id'] = param_id

    # Add default prototype schedules when prototype mode is enabled but no prototype was specified
    # This happens when loading from CSV, since the param_id only encodes the factor, not the prototype
    if params.get('gamma_is_prototype') and 'gamma_prototype' not in params:
        params['gamma_prototype'] = "lin(0,1)"  # Default prototype for gamma
    if params.get('beta_is_prototype') and 'beta_prototype' not in params:
        params['beta_prototype'] = "lin(0,1)"  # Default prototype for beta

    return params

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
                elif any(token in val for token in ['mu_max', 'alpha', 'num_vertices', 'num_edges', 'num_states', 'gamma_th', 'T', 'dt']):
                    # This is an expression to be evaluated later with environment variables
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

def generate_param_sets(model_type, model_params, T, dt, hyperparams_from_csv=None):
    """
    Generate all parameter combinations for a model type.

    Parameters:
    - model_type: ModelType enum
    - model_params: Dictionary of model parameters from config
    - T: Simulation time
    - dt: Time step
    - hyperparams_from_csv: Optional dictionary of hyperparameters from hyperparam table to merge with config params

    Returns:
    - List of parameter sets (dictionaries)
    
    Precedence (highest to lowest):
    1. Model config (explicit values in YAML)
    2. Hyperparam table (if provided)
    3. Defaults (hardcoded fallbacks)
    """
    # Handle case where model_params is None (YAML interprets "MODEL:" with nothing as None)
    if model_params is None:
        model_params = {}

    num_steps = int(np.floor(T / dt))
    
    # Helper function to get parameter with proper precedence:
    # Config > Hyperparam table > Default
    def get_param(param_name, default):
        """Get parameter value with precedence: config > hyperparam table > default"""
        # First check if explicitly set in config
        if param_name in model_params and model_params[param_name] is not None:
            return model_params[param_name]
        # Then check hyperparam table
        if hyperparams_from_csv is not None and param_name in hyperparams_from_csv:
            return hyperparams_from_csv[param_name]
        # Finally use default
        return default
    
    # Common function to process schedule definitions including linked and prototype schedules
    def process_schedule_param(param_name, default_value):
        # Check if this is a linked or prototype schedule
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
            # This schedule is based on another without a factor - factor will come from hyperparam table
            elif 'based_on' in schedule_def and 'factor' not in schedule_def:
                base_schedule = schedule_def['based_on']
                return {
                    'type': 'linked_only',
                    'base_schedule': base_schedule
                }
            # This is a prototype schedule with a factor
            elif 'prototype' in schedule_def and 'factor' in schedule_def:
                prototype = schedule_def['prototype']
                factors = expand_param_values(schedule_def['factor'])
                return {
                    'type': 'prototype',
                    'prototype': prototype,
                    'factors': factors
                }
            # This is a prototype schedule without a factor - factor will come from hyperparam table
            elif 'prototype' in schedule_def and 'factor' not in schedule_def:
                prototype = schedule_def['prototype']
                return {
                    'type': 'prototype_only',
                    'prototype': prototype
                }
            # Handle start/span specification
            elif 'start' in schedule_def and 'span' in schedule_def:
                # Validate no conflicting keys
                if 'based_on' in schedule_def or 'prototype' in schedule_def or 'factor' in schedule_def:
                    raise ValueError(
                        f"{param_name}: Cannot use 'start'/'span' with 'based_on'/'prototype'/'factor'"
                    )

                start_expr = schedule_def['start']
                span_values = expand_param_values(schedule_def['span'])

                return {
                    'type': 'linspan',
                    'start_expr': start_expr,
                    'span_values': span_values
                }
            else:
                raise ValueError(f"Invalid schedule definition for {param_name}")
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
    if model_type == ModelType.POLYNOMIAL or model_type == ModelType.QPDC:
        # Handle expression-based poly_order
        raw_poly_orders = expand_param_values(model_params.get('poly_order', [3]))
        poly_orders = []
        for po in raw_poly_orders:
            if isinstance(po, str) and any(token in po for token in ['num_states', 'num_vertices', 'num_edges']):
                # Will be evaluated later with environment variables
                poly_orders.append(po)
            else:
                poly_orders.append(int(po))
        
        # Process schedules
        beta_info = process_schedule_param('beta_schedule', ["lin(0,1)"])
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,1)"])
        
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
        
        # Handle gamma as prototype with factors
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'prototype':
            for poly_order, beta_expr, factor in itertools.product(
                    poly_orders, beta_info['schedules'], gamma_info['factors']):
                param_id = f"po{poly_order}_b{beta_expr}_gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True
                })
        
        # Handle beta as prototype with factors
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'prototype':
            for poly_order, gamma_expr, factor in itertools.product(
                    poly_orders, gamma_info['schedules'], beta_info['factors']):
                param_id = f"po{poly_order}_g{gamma_expr}_bpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_factor': factor,
                    'beta_is_prototype': True
                })

        # Handle gamma based on beta without factor (factor will come from hyperparam table)
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linked_only' and gamma_info['base_schedule'] == 'beta_schedule':
            for poly_order, beta_expr in itertools.product(poly_orders, beta_info['schedules']):
                param_id = f"po{poly_order}_b{beta_expr}_gf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_based_on': 'beta'
                })

        # Handle beta based on gamma without factor (factor will come from hyperparam table)
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'linked_only' and beta_info['base_schedule'] == 'gamma_schedule':
            for poly_order, gamma_expr in itertools.product(poly_orders, gamma_info['schedules']):
                param_id = f"po{poly_order}_g{gamma_expr}_bf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_based_on': 'gamma'
                })

        # Handle gamma as prototype without factor (factor will come from hyperparam table)
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'prototype_only':
            for poly_order, beta_expr in itertools.product(poly_orders, beta_info['schedules']):
                param_id = f"po{poly_order}_b{beta_expr}_gpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_is_prototype': True
                })

        # Handle beta as prototype without factor (factor will come from hyperparam table)
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'prototype_only':
            for poly_order, gamma_expr in itertools.product(poly_orders, gamma_info['schedules']):
                param_id = f"po{poly_order}_g{gamma_expr}_bpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_is_prototype': True
                })

        # Handle beta as linspan
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'direct':
            for poly_order, span, gamma_expr in itertools.product(
                    poly_orders, beta_info['span_values'], gamma_info['schedules']):
                beta_expr = f"linspan({beta_info['start_expr']},{span})"
                param_id = f"po{poly_order}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle gamma as linspan
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linspan':
            for poly_order, beta_expr, span in itertools.product(
                    poly_orders, beta_info['schedules'], gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{span})"
                param_id = f"po{poly_order}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle both as linspan
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linspan':
            for poly_order, beta_span, gamma_span in itertools.product(
                    poly_orders, beta_info['span_values'], gamma_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"po{poly_order}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle beta as linspan with gamma linked
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linked' and gamma_info['base_schedule'] == 'beta_schedule':
            for poly_order, beta_span, factor in itertools.product(
                    poly_orders, beta_info['span_values'], gamma_info['factors']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"po{poly_order}_b{beta_expr}_gf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_factor': factor,
                    'gamma_based_on': 'beta'
                })

        # Handle gamma as linspan with beta linked
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'linked' and beta_info['base_schedule'] == 'gamma_schedule':
            for poly_order, gamma_span, factor in itertools.product(
                    poly_orders, gamma_info['span_values'], beta_info['factors']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"po{poly_order}_g{gamma_expr}_bf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_factor': factor,
                    'beta_based_on': 'gamma'
                })

        # Handle beta as linspan with gamma prototype
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'prototype':
            for poly_order, beta_span, factor in itertools.product(
                    poly_orders, beta_info['span_values'], gamma_info['factors']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"po{poly_order}_b{beta_expr}_gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True
                })

        # Handle gamma as linspan with beta prototype
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'prototype':
            for poly_order, gamma_span, factor in itertools.product(
                    poly_orders, gamma_info['span_values'], beta_info['factors']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"po{poly_order}_g{gamma_expr}_bpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_factor': factor,
                    'beta_is_prototype': True
                })

        # Handle beta as linspan with gamma linked_only (factor from hyperparam table)
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linked_only' and gamma_info['base_schedule'] == 'beta_schedule':
            for poly_order, beta_span in itertools.product(poly_orders, beta_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"po{poly_order}_b{beta_expr}_gf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_based_on': 'beta'
                })

        # Handle gamma as linspan with beta linked_only (factor from hyperparam table)
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'linked_only' and beta_info['base_schedule'] == 'gamma_schedule':
            for poly_order, gamma_span in itertools.product(poly_orders, gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"po{poly_order}_g{gamma_expr}_bf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_based_on': 'gamma'
                })

        # Handle beta as linspan with gamma prototype_only (factor from hyperparam table)
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'prototype_only':
            for poly_order, beta_span in itertools.product(poly_orders, beta_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"po{poly_order}_b{beta_expr}_gpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_is_prototype': True
                })

        # Handle gamma as linspan with beta prototype_only (factor from hyperparam table)
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'prototype_only':
            for poly_order, gamma_span in itertools.product(poly_orders, gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"po{poly_order}_g{gamma_expr}_bpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_is_prototype': True
                })

        else:
            raise ValueError("Invalid schedule linkage configuration")

    elif model_type == ModelType.NEC:
        # Expand all parameter ranges using get_param for proper precedence
        poly_orders = [int(po) for po in expand_param_values(get_param('poly_order', [3]))]
        alpha_rates = expand_param_values(get_param('alpha_rate', [1e-2]))
        r_targets = expand_param_values(get_param('r_target', [2.0]))
        initial_alphas = expand_param_values(get_param('initial_alpha', [1.0]))
        
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,1)"])

        param_sets = []

        # Handle the case where gamma is directly specified
        if gamma_info['type'] == 'direct':
            for poly_order, alpha_rate, r_target, initial_alpha, gamma_expr in itertools.product(
                    poly_orders, alpha_rates, r_targets, initial_alphas, gamma_info['schedules']):
                param_id = f"ar{alpha_rate:.2e}_rt{r_target}_ia{initial_alpha}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'alpha_rate': alpha_rate,
                    'gamma_expr': gamma_expr,
                    'r_target': r_target,
                    'initial_alpha': initial_alpha,
                    'initial_alpha_expr': initial_alpha if isinstance(initial_alpha, str) else None
                })

        # Handle gamma as prototype with factors
        elif gamma_info['type'] == 'prototype':
            for poly_order, alpha_rate, r_target, initial_alpha, factor in itertools.product(
                    poly_orders, alpha_rates, r_targets, initial_alphas, gamma_info['factors']):
                param_id = f"ar{alpha_rate:.2e}_rt{r_target}_ia{initial_alpha}_gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'alpha_rate': alpha_rate,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True,
                    'r_target': r_target,
                    'initial_alpha': initial_alpha,
                    'initial_alpha_expr': initial_alpha if isinstance(initial_alpha, str) else None
                })

        # Handle gamma as prototype only (factor will come from hyperparam table)
        elif gamma_info['type'] == 'prototype_only':
            for poly_order, alpha_rate, r_target, initial_alpha in itertools.product(
                    poly_orders, alpha_rates, r_targets, initial_alphas):
                # Param ID will be updated when factor is filled from hyperparam table
                param_id = f"ar{alpha_rate:.2e}_rt{r_target}_ia{initial_alpha}_gpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'alpha_rate': alpha_rate,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_is_prototype': True,
                    'r_target': r_target,
                    'initial_alpha': initial_alpha,
                    'initial_alpha_expr': initial_alpha if isinstance(initial_alpha, str) else None
                })

        # Handle gamma as linspan
        elif gamma_info['type'] == 'linspan':
            for poly_order, alpha_rate, r_target, initial_alpha, span in itertools.product(
                    poly_orders, alpha_rates, r_targets, initial_alphas, gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{span})"
                param_id = f"ar{alpha_rate:.2e}_rt{r_target}_ia{initial_alpha}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'poly_order': poly_order,
                    'alpha_rate': alpha_rate,
                    'gamma_expr': gamma_expr,
                    'r_target': r_target,
                    'initial_alpha': initial_alpha,
                    'initial_alpha_expr': initial_alpha if isinstance(initial_alpha, str) else None
                })

        else:
            raise ValueError("NEC model only supports direct, prototype, prototype_only, or linspan gamma_schedule specification")

    elif model_type == ModelType.SIGMOID:
        # Expand scalar parameters using get_param for proper precedence
        alphas = expand_param_values(get_param('alpha', [-1.0]))
        
        # Process schedules
        beta_info = process_schedule_param('beta_schedule', ["lin(0,1)"])
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,1)"])
        
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
    
        # Handle gamma as prototype with factors
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'prototype':
            for alpha, beta_expr, factor in itertools.product(
                    alphas, beta_info['schedules'], gamma_info['factors']):
                param_id = f"a{alpha}_b{beta_expr}_gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True
                })
    
        # Handle beta as prototype with factors
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'prototype':
            for alpha, gamma_expr, factor in itertools.product(
                    alphas, gamma_info['schedules'], beta_info['factors']):
                param_id = f"a{alpha}_g{gamma_expr}_bpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_factor': factor,
                    'beta_is_prototype': True
                })

        # Handle gamma based on beta without factor (factor will come from hyperparam table)
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linked_only' and gamma_info['base_schedule'] == 'beta_schedule':
            for alpha, beta_expr in itertools.product(alphas, beta_info['schedules']):
                param_id = f"a{alpha}_b{beta_expr}_gf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_based_on': 'beta'
                })

        # Handle beta based on gamma without factor (factor will come from hyperparam table)
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'linked_only' and beta_info['base_schedule'] == 'gamma_schedule':
            for alpha, gamma_expr in itertools.product(alphas, gamma_info['schedules']):
                param_id = f"a{alpha}_g{gamma_expr}_bf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_based_on': 'gamma'
                })

        # Handle gamma as prototype without factor (factor will come from hyperparam table)
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'prototype_only':
            for alpha, beta_expr in itertools.product(alphas, beta_info['schedules']):
                param_id = f"a{alpha}_b{beta_expr}_gpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_is_prototype': True
                })

        # Handle beta as prototype without factor (factor will come from hyperparam table)
        elif gamma_info['type'] == 'direct' and beta_info['type'] == 'prototype_only':
            for alpha, gamma_expr in itertools.product(alphas, gamma_info['schedules']):
                param_id = f"a{alpha}_g{gamma_expr}_bpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_is_prototype': True
                })

        # Handle beta as linspan
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'direct':
            for alpha, span, gamma_expr in itertools.product(
                    alphas, beta_info['span_values'], gamma_info['schedules']):
                beta_expr = f"linspan({beta_info['start_expr']},{span})"
                param_id = f"a{alpha}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle gamma as linspan
        elif beta_info['type'] == 'direct' and gamma_info['type'] == 'linspan':
            for alpha, beta_expr, span in itertools.product(
                    alphas, beta_info['schedules'], gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{span})"
                param_id = f"a{alpha}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle both as linspan
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linspan':
            for alpha, beta_span, gamma_span in itertools.product(
                    alphas, beta_info['span_values'], gamma_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"a{alpha}_b{beta_expr}_g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_expr': gamma_expr
                })

        # Handle beta as linspan with gamma linked
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linked' and gamma_info['base_schedule'] == 'beta_schedule':
            for alpha, beta_span, factor in itertools.product(
                    alphas, beta_info['span_values'], gamma_info['factors']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"a{alpha}_b{beta_expr}_gf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_factor': factor,
                    'gamma_based_on': 'beta'
                })

        # Handle gamma as linspan with beta linked
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'linked' and beta_info['base_schedule'] == 'gamma_schedule':
            for alpha, gamma_span, factor in itertools.product(
                    alphas, gamma_info['span_values'], beta_info['factors']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"a{alpha}_g{gamma_expr}_bf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_factor': factor,
                    'beta_based_on': 'gamma'
                })

        # Handle beta as linspan with gamma prototype
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'prototype':
            for alpha, beta_span, factor in itertools.product(
                    alphas, beta_info['span_values'], gamma_info['factors']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"a{alpha}_b{beta_expr}_gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True
                })

        # Handle gamma as linspan with beta prototype
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'prototype':
            for alpha, gamma_span, factor in itertools.product(
                    alphas, gamma_info['span_values'], beta_info['factors']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"a{alpha}_g{gamma_expr}_bpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_factor': factor,
                    'beta_is_prototype': True
                })

        # Handle beta as linspan with gamma linked_only (factor from hyperparam table)
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'linked_only' and gamma_info['base_schedule'] == 'beta_schedule':
            for alpha, beta_span in itertools.product(alphas, beta_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"a{alpha}_b{beta_expr}_gf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_based_on': 'beta'
                })

        # Handle gamma as linspan with beta linked_only (factor from hyperparam table)
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'linked_only' and beta_info['base_schedule'] == 'gamma_schedule':
            for alpha, gamma_span in itertools.product(alphas, gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"a{alpha}_g{gamma_expr}_bf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_based_on': 'gamma'
                })

        # Handle beta as linspan with gamma prototype_only (factor from hyperparam table)
        elif beta_info['type'] == 'linspan' and gamma_info['type'] == 'prototype_only':
            for alpha, beta_span in itertools.product(alphas, beta_info['span_values']):
                beta_expr = f"linspan({beta_info['start_expr']},{beta_span})"
                param_id = f"a{alpha}_b{beta_expr}_gpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'beta_expr': beta_expr,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_is_prototype': True
                })

        # Handle gamma as linspan with beta prototype_only (factor from hyperparam table)
        elif gamma_info['type'] == 'linspan' and beta_info['type'] == 'prototype_only':
            for alpha, gamma_span in itertools.product(alphas, gamma_info['span_values']):
                gamma_expr = f"linspan({gamma_info['start_expr']},{gamma_span})"
                param_id = f"a{alpha}_g{gamma_expr}_bpf_TBD"
                param_sets.append({
                    'id': param_id,
                    'alpha': alpha,
                    'gamma_expr': gamma_expr,
                    'beta_prototype': beta_info['prototype'],
                    'beta_is_prototype': True
                })

        else:
            raise ValueError("Invalid schedule linkage configuration")

    elif model_type == ModelType.FIXED_AMPLITUDE:
        # Process gamma schedule
        gamma_info = process_schedule_param('gamma_schedule', ["lin(0,1)"])
        
        param_sets = []

        # Handle the case where gamma is directly specified
        if gamma_info['type'] == 'direct':
            for gamma_expr in gamma_info['schedules']:
                param_id = f"g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'gamma_expr': gamma_expr
                })

        # Handle gamma as prototype with factors
        elif gamma_info['type'] == 'prototype':
            for factor in gamma_info['factors']:
                param_id = f"gpf{factor}"
                param_sets.append({
                    'id': param_id,
                    'gamma_prototype': gamma_info['prototype'],
                    'gamma_factor': factor,
                    'gamma_is_prototype': True
                })

        # Handle gamma as prototype without factor (factor will come from hyperparam table)
        elif gamma_info['type'] == 'prototype_only':
            param_id = f"gpf_TBD"
            param_sets.append({
                'id': param_id,
                'gamma_prototype': gamma_info['prototype'],
                'gamma_is_prototype': True
            })

        # Handle gamma as linspan
        elif gamma_info['type'] == 'linspan':
            for span in gamma_info['span_values']:
                gamma_expr = f"linspan({gamma_info['start_expr']},{span})"
                param_id = f"g{gamma_expr}"
                param_sets.append({
                    'id': param_id,
                    'gamma_expr': gamma_expr
                })

        else:
            raise ValueError("FIXED_AMPLITUDE model only supports direct, prototype, prototype_only, or linspan gamma_schedule specification")

    elif model_type == ModelType.CIM:
        # Expand all parameter ranges for CIM model using get_param for proper precedence
        alphas = expand_param_values(get_param('alpha', [-10.0]))
        
        # Check for B_num_vertices (scaled B) or regular B
        # Check config first, then hyperparam table, then default to regular B
        if 'B_num_vertices' in model_params:
            B_num_vertices_values = expand_param_values(model_params.get('B_num_vertices', [18]))
            use_scaled_B = True
        elif hyperparams_from_csv is not None and 'B_num_vertices' in hyperparams_from_csv:
            B_num_vertices_values = expand_param_values(hyperparams_from_csv['B_num_vertices'])
            use_scaled_B = True
        elif 'B' in model_params:
            Bs = expand_param_values(model_params.get('B', [18/100]))
            use_scaled_B = False
        elif hyperparams_from_csv is not None and 'B' in hyperparams_from_csv:
            Bs = expand_param_values(hyperparams_from_csv['B'])
            use_scaled_B = False
        else:
            Bs = expand_param_values([18/100])
            use_scaled_B = False
            
        zetas = expand_param_values(get_param('zeta', [0.6]))
        
        beta_info = process_schedule_param('beta_schedule', ["lin(0,0.01)"])
        
        param_sets = []
        
        # Handle the case where beta is directly specified
        if beta_info['type'] == 'direct':
            if use_scaled_B:
                # Use B_num_vertices parameter
                for alpha, B_num_vertices, zeta, beta_expr in itertools.product(
                        alphas, B_num_vertices_values, zetas, beta_info['schedules']):
                    param_id = f"a{alpha}_Bnv{B_num_vertices}_z{zeta}_b{beta_expr}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B_num_vertices': B_num_vertices,
                        'zeta': zeta,
                        'beta_expr': beta_expr
                    })
            else:
                # Use regular B parameter
                for alpha, B, zeta, beta_expr in itertools.product(
                        alphas, Bs, zetas, beta_info['schedules']):
                    param_id = f"a{alpha}_B{B}_z{zeta}_b{beta_expr}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B': B,
                        'zeta': zeta,
                        'beta_expr': beta_expr
                    })
        
        # Handle beta as prototype with factors
        elif beta_info['type'] == 'prototype':
            if use_scaled_B:
                # Use B_num_vertices parameter
                for alpha, B_num_vertices, zeta, factor in itertools.product(
                        alphas, B_num_vertices_values, zetas, beta_info['factors']):
                    param_id = f"a{alpha}_Bnv{B_num_vertices}_z{zeta}_bpf{factor}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B_num_vertices': B_num_vertices,
                        'zeta': zeta,
                        'beta_prototype': beta_info['prototype'],
                        'beta_factor': factor,
                        'beta_is_prototype': True
                    })
            else:
                # Use regular B parameter
                for alpha, B, zeta, factor in itertools.product(
                        alphas, Bs, zetas, beta_info['factors']):
                    param_id = f"a{alpha}_B{B}_z{zeta}_bpf{factor}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B': B,
                        'zeta': zeta,
                        'beta_prototype': beta_info['prototype'],
                        'beta_factor': factor,
                        'beta_is_prototype': True
                    })

        # Handle beta as prototype without factor (factor will come from hyperparam table)
        elif beta_info['type'] == 'prototype_only':
            if use_scaled_B:
                # Use B_num_vertices parameter
                for alpha, B_num_vertices, zeta in itertools.product(
                        alphas, B_num_vertices_values, zetas):
                    param_id = f"a{alpha}_Bnv{B_num_vertices}_z{zeta}_bpf_TBD"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B_num_vertices': B_num_vertices,
                        'zeta': zeta,
                        'beta_prototype': beta_info['prototype'],
                        'beta_is_prototype': True
                    })
            else:
                # Use regular B parameter
                for alpha, B, zeta in itertools.product(
                        alphas, Bs, zetas):
                    param_id = f"a{alpha}_B{B}_z{zeta}_bpf_TBD"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B': B,
                        'zeta': zeta,
                        'beta_prototype': beta_info['prototype'],
                        'beta_is_prototype': True
                    })

        # Handle beta as linspan
        elif beta_info['type'] == 'linspan':
            if use_scaled_B:
                for alpha, B_num_vertices, zeta, span in itertools.product(
                        alphas, B_num_vertices_values, zetas, beta_info['span_values']):
                    beta_expr = f"linspan({beta_info['start_expr']},{span})"
                    param_id = f"a{alpha}_Bnv{B_num_vertices}_z{zeta}_b{beta_expr}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B_num_vertices': B_num_vertices,
                        'zeta': zeta,
                        'beta_expr': beta_expr
                    })
            else:
                for alpha, B, zeta, span in itertools.product(
                        alphas, Bs, zetas, beta_info['span_values']):
                    beta_expr = f"linspan({beta_info['start_expr']},{span})"
                    param_id = f"a{alpha}_B{B}_z{zeta}_b{beta_expr}"
                    param_sets.append({
                        'id': param_id,
                        'alpha': alpha,
                        'B': B,
                        'zeta': zeta,
                        'beta_expr': beta_expr
                    })

        else:
            raise ValueError("CIM model only supports direct, prototype, prototype_only, or linspan beta_schedule specification")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # If hyperparam table provided, merge with each param_set
    # Hyperparam table values should fill in any parameters that weren't explicitly set in the config
    # Note: Some parameters (like r_target, B_num_vertices, zeta) may already have been filled
    # by the get_param() helper above, so this mainly handles schedule factors and other params
    if hyperparams_from_csv is not None:
        # Determine which parameters were explicitly set in model_params
        explicit_params = set()
        for key, value in model_params.items():
            if value is not None and value != {} and value != []:
                explicit_params.add(key)
                # Also mark related keys as explicit
                if key == 'gamma_schedule':
                    explicit_params.add('gamma_expr')
                    # Only mark gamma_factor as explicit if factor was specified in the schedule
                    if isinstance(value, dict) and 'factor' in value:
                        explicit_params.add('gamma_factor')
                if key == 'beta_schedule':
                    explicit_params.add('beta_expr')
                    # Only mark beta_factor as explicit if factor was specified in the schedule
                    if isinstance(value, dict) and 'factor' in value:
                        explicit_params.add('beta_factor')

        for param_set in param_sets:
            # Fill in parameters from hyperparam table that weren't explicitly set in config
            for key, value in hyperparams_from_csv.items():
                # Skip the 'id' field from CSV - we manage our own IDs
                if key == 'id':
                    continue
                    
                # Skip if this parameter was explicitly set in config
                if key in explicit_params:
                    continue

                # For expressions and factors, check parent parameter too
                if key == 'gamma_expr' and 'gamma_schedule' in explicit_params:
                    continue
                if key == 'gamma_factor' and 'gamma_schedule' in explicit_params and 'gamma_factor' in explicit_params:
                    continue
                if key == 'beta_expr' and 'beta_schedule' in explicit_params:
                    continue
                if key == 'beta_factor' and 'beta_schedule' in explicit_params and 'beta_factor' in explicit_params:
                    continue

                # Use hyperparam table value (only if not already set by get_param earlier)
                if key not in param_set:
                    param_set[key] = value

            # Update param_id if we filled in factors from hyperparam table
            # Replace _TBD placeholders with actual values
            if '_TBD' in param_set.get('id', ''):
                new_id = param_set['id']
                if 'gpf_TBD' in new_id and 'gamma_factor' in param_set:
                    new_id = new_id.replace('gpf_TBD', f"gpf{param_set['gamma_factor']}")
                if 'bpf_TBD' in new_id and 'beta_factor' in param_set:
                    new_id = new_id.replace('bpf_TBD', f"bpf{param_set['beta_factor']}")
                if 'gf_TBD' in new_id and 'gamma_factor' in param_set:
                    new_id = new_id.replace('gf_TBD', f"gf{param_set['gamma_factor']}")
                if 'bf_TBD' in new_id and 'beta_factor' in param_set:
                    new_id = new_id.replace('bf_TBD', f"bf{param_set['beta_factor']}")
                param_set['id'] = new_id

    return param_sets

def run_task(graph, model_type, param_set, seed, T, dt, num_states, noise_factor):
    """Run a single task with the given parameters"""
    num_vertices = graph['num_vertices']
    num_edges = graph['num_edges']
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
        'np': np,
        'num_vertices': num_vertices,
        'num_edges': num_edges,
        'num_states': num_states,
        'gamma_th': (256/27)**(1/4),
        'T': T,
        'dt': dt
    }
    
    # Add model-specific parameters to environment
    for key, value in param_set.items():
        if key not in ['id', 'beta_fn', 'gamma_fn', 'beta_factor', 'gamma_factor', 'beta_based_on', 'gamma_based_on',
                       'beta_is_prototype', 'gamma_is_prototype', 'beta_prototype', 'gamma_prototype']:
            env[key] = value
    
    # Evaluate poly_order if it's an expression
    if 'poly_order' in param_set and isinstance(param_set['poly_order'], str):
        param_set['poly_order'] = int(safe_eval(param_set['poly_order'], env))

    # Evaluate factor expressions if they're strings
    if 'gamma_factor' in param_set and isinstance(param_set['gamma_factor'], str):
        param_set['gamma_factor'] = safe_eval(param_set['gamma_factor'], env)
    if 'beta_factor' in param_set and isinstance(param_set['beta_factor'], str):
        param_set['beta_factor'] = safe_eval(param_set['beta_factor'], env)

    # Evaluate schedules based on model type requirements
    if model_type == ModelType.POLYNOMIAL or model_type == ModelType.QPDC:
        # For POLYNOMIAL model
        if 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            # gamma = factor * beta
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * b for b in beta_schedule]
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            # beta = factor * gamma
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * g for g in gamma_schedule]
        elif 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            # gamma = prototype * factor
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
        elif 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            # beta = prototype * factor
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
        else:
            # Both schedules directly defined
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
        
        seed_amplitude = 0
        if model_type == ModelType.QPDC:
            # For QPDC model, we need to set the seed amplitude
            seed_amplitude = 1

        res = potts_sim.run_polynomial(
            T, dt, num_vertices, num_states,
            edges,
            noise_factor, task_seed,
            seed_amplitude,
            param_set['poly_order'],
            beta_schedule, gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True,
            return_last_only=False
        )
    
    elif model_type == ModelType.NEC:
        # For NEC model
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            # gamma = prototype * factor
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
        else:
            # gamma schedule is directly defined
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)

        # Handle special case for initial_alpha if it's an expression
        if 'initial_alpha_expr' in param_set and param_set['initial_alpha_expr']:
            # Evaluate the expression (e.g., "mu_max * -1")
            initial_alpha = safe_eval(param_set['initial_alpha_expr'], env)
        else:
            initial_alpha = param_set['initial_alpha']
            
        initial_alpha_arr = [initial_alpha] * num_vertices
        
        res = potts_sim.run_nec(
            T, dt, num_vertices, num_states,
            edges,
            noise_factor, task_seed,
            param_set['poly_order'],
            param_set['alpha_rate'], param_set['r_target'],
            initial_alpha_arr, gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True,
            return_last_only=False
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
        elif 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            # gamma = prototype * factor
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
        elif 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            # beta = prototype * factor
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
        else:
            # Both schedules directly defined
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            
        res = potts_sim.run_sigmoid(
            T, dt, num_vertices, num_states,
            edges,
            noise_factor, task_seed,
            param_set['alpha'],
            beta_schedule, gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True,
            return_last_only=False
        )
    
    elif model_type == ModelType.FIXED_AMPLITUDE:
        # For FIXED_AMPLITUDE model
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            # gamma = prototype * factor
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
        else:
            # gamma schedule is directly defined
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
        
        res = potts_sim.run_fixed_amplitude(
            T, dt, num_vertices, num_states,
            edges,
            noise_factor, task_seed,
            gamma_schedule,
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True,
            return_last_only=False
        )
    
    elif model_type == ModelType.CIM:
        # For CIM model
        # Evaluate beta schedule
        if 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            # beta = prototype * factor
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
        else:
            # beta schedule is directly defined
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
        
        # Determine B value - either directly specified or calculated from B_num_vertices
        if 'B_num_vertices' in param_set:
            # Calculate B = B_num_vertices / num_vertices
            B = param_set['B_num_vertices'] / num_vertices
        else:
            # Use B directly
            B = param_set['B']
        
        res = run_cim_from_graph(
            T, dt, num_vertices, num_states,
            edges,
            noise_factor, task_seed,
            param_set['alpha'],
            beta_schedule,
            B, param_set['zeta'],
            return_continuous_states=False,
            return_discrete_states=False,
            return_energy=True,
            return_cut_value=True,
            return_best_only=True,
            return_last_only=False
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract results
    best_cut = float(res["cut_value"][0])
    best_energy = float(res["energy"][0])
    best_step = int(res["step"])
    elapsed_time = time.time() - start_time

    # Number of spins is the number of vertices in the graph except for CIM
    num_spins = num_vertices if model_type != ModelType.CIM else res['num_spins']

    # Optimum cut value and energy from graph
    opt_cut = graph['opt_cut_dict'].get(num_states)
    opt_energy = graph['opt_energy_dict'].get(num_states)
    
    # Return results in a dictionary
    result = {
        "graph": graph['name'],
        "num_vertices": graph['num_vertices'],
        "num_spins": num_spins,
        "num_edges": num_edges,
        "model": model_type.name,
        "param_id": param_set['id'],
        "seed": seed,
        "cut_value": best_cut,
        "energy": best_energy,
        "step": best_step,
        "opt_cut": opt_cut,
        "opt_energy": opt_energy,
        "cut_gap": (best_cut - opt_cut) if opt_cut is not None else None,
        "energy_gap": (best_energy - opt_energy) if opt_energy is not None else None,
        "runtime": elapsed_time,
        "T": T,
        "dt": dt,
        "num_steps": num_steps,
        "mu_max": mu_max
    }
    
    # Add model-specific parameters to the result
    if model_type == ModelType.POLYNOMIAL or model_type == ModelType.QPDC:
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
        elif 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_prototype": param_set['gamma_prototype'],
                "gamma_factor": param_set['gamma_factor']
            })
        elif 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            result.update({
                "gamma_schedule": param_set['gamma_expr'],
                "beta_prototype": param_set['beta_prototype'],
                "beta_factor": param_set['beta_factor']
            })
        else:
            # Both schedules directly defined
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_schedule": param_set['gamma_expr']
            })
    elif model_type == ModelType.NEC:
        result.update({
            "poly_order": param_set['poly_order'],
            "alpha_rate": param_set['alpha_rate'],
            "r_target": param_set['r_target'],
            "initial_alpha": param_set['initial_alpha']
        })
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            result.update({
                "gamma_prototype": param_set['gamma_prototype'],
                "gamma_factor": param_set['gamma_factor']
            })
        else:
            # Directly specified gamma schedule
            result.update({
                "gamma_schedule": param_set['gamma_expr']
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
        elif 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_prototype": param_set['gamma_prototype'],
                "gamma_factor": param_set['gamma_factor']
            })
        elif 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            result.update({
                "gamma_schedule": param_set['gamma_expr'],
                "beta_prototype": param_set['beta_prototype'],
                "beta_factor": param_set['beta_factor']
            })
        else:
            # Both schedules directly defined
            result.update({
                "beta_schedule": param_set['beta_expr'],
                "gamma_schedule": param_set['gamma_expr']
            })
    elif model_type == ModelType.FIXED_AMPLITUDE:
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            result.update({
                "gamma_prototype": param_set['gamma_prototype'],
                "gamma_factor": param_set['gamma_factor']
            })
        else:
            # Directly specified gamma schedule
            result.update({
                "gamma_schedule": param_set['gamma_expr']
            })
    elif model_type == ModelType.CIM:
        result.update({
            'alpha': param_set['alpha'],
            'zeta': param_set['zeta']
        })
        
        # Add B parameter info (either direct or scaled)
        if 'B_num_vertices' in param_set:
            B = param_set['B_num_vertices'] / num_vertices
            result.update({
                'B_num_vertices': param_set['B_num_vertices'],
                'B': B
            })
        else:
            result.update({
                'B': param_set['B']
            })
        
        # Add beta schedule information
        if 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            result.update({
                'beta_prototype': param_set['beta_prototype'],
                'beta_factor': param_set['beta_factor']
            })
        else:
            result.update({
                'beta_schedule': param_set['beta_expr']
            })
    
    return result

def sanitize_filename(name):
    """Replace characters that are invalid in filenames with underscores"""
    # Replace characters that are invalid in filenames
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def visualize_schedules(graph, model_type, param_set, T, dt, out_dir):
    """Generate visualizations of schedules for a parameter set"""
    import matplotlib.pyplot as plt # Only import matplotlib if needed
    num_steps = int(np.floor(T / dt))
    mu_max = graph['mu_max']
    
    # Prepare environment for schedule evaluation
    env = {
        'mu_max': mu_max,
        'np': np,
        'num_vertices': graph['num_vertices'],
        'num_edges': graph['num_edges'],
        'num_states': param_set.get('num_states', 3),
        'gamma_th': (256/27)**(1/4)  # Add gamma_th constant
    }
    
    # Add model-specific parameters to environment
    for key, value in param_set.items():
        if key not in ['id', 'beta_fn', 'gamma_fn', 'beta_factor', 'gamma_factor', 'beta_based_on', 'gamma_based_on',
                       'beta_is_prototype', 'gamma_is_prototype', 'beta_prototype', 'gamma_prototype']:
            env[key] = value
    
    # Evaluate poly_order if it's an expression
    if 'poly_order' in param_set and isinstance(param_set['poly_order'], str):
        try:
            param_set['poly_order'] = int(safe_eval(param_set['poly_order'], env))
        except:
            # Keep as string if evaluation fails
            pass
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.linspace(0, T, num_steps)
    
    # Evaluate schedules based on model type
    if model_type == ModelType.POLYNOMIAL or model_type == ModelType.QPDC:
        # Evaluate beta schedule
        if 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
            beta_label = f"Beta (prototype={param_set['beta_prototype']}, factor={param_set['beta_factor']})"
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * g for g in gamma_schedule]
            beta_label = f"Beta (based on gamma, factor={param_set['beta_factor']})"
        else:
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            beta_label = f"Beta ({param_set['beta_expr']})"
        
        # Evaluate gamma schedule
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
            gamma_label = f"Gamma (prototype={param_set['gamma_prototype']}, factor={param_set['gamma_factor']})"
        elif 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            gamma_schedule = [param_set['gamma_factor'] * b for b in beta_schedule]
            gamma_label = f"Gamma (based on beta, factor={param_set['gamma_factor']})"
        else:
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            gamma_label = f"Gamma ({param_set['gamma_expr']})"
        
        # Plot schedules
        ax.plot(time_axis, beta_schedule, label=beta_label)
        ax.plot(time_axis, gamma_schedule, label=gamma_label)
        ax.set_title(f"{model_type.name} Model - Polynomial Order: {param_set['poly_order']}")
        
    elif model_type == ModelType.NEC:
        # For NEC model
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
            gamma_label = f"Gamma (prototype={param_set['gamma_prototype']}, factor={param_set['gamma_factor']})"
        else:
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            gamma_label = f"Gamma ({param_set['gamma_expr']})"
        
        # Handle special case for initial_alpha if it's an expression
        if 'initial_alpha_expr' in param_set and param_set['initial_alpha_expr']:
            initial_alpha = safe_eval(param_set['initial_alpha_expr'], env)
            alpha_label = f"Initial Alpha: {param_set['initial_alpha_expr']} = {initial_alpha}"
        else:
            initial_alpha = param_set['initial_alpha']
            alpha_label = f"Initial Alpha: {initial_alpha}"
        
        # Plot gamma schedule
        ax.plot(time_axis, gamma_schedule, label=gamma_label)

        # Add alpha rate and r_target as text annotations
        alpha_rate_text = f"Alpha Rate: {param_set['alpha_rate']}"
        r_target_text = f"R Target: {param_set['r_target']}"
        
        ax.text(0.02, 0.95, alpha_label, transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.90, alpha_rate_text, transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.85, r_target_text, transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f"NEC Model - Polynomial Order: {param_set['poly_order']}")
        
    elif model_type == ModelType.SIGMOID:
        # Evaluate beta schedule
        if 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
            beta_label = f"Beta (prototype={param_set['beta_prototype']}, factor={param_set['beta_factor']})"
        elif 'beta_based_on' in param_set and param_set['beta_based_on'] == 'gamma':
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * g for g in gamma_schedule]
            beta_label = f"Beta (based on gamma, factor={param_set['beta_factor']})"
        else:
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            beta_label = f"Beta ({param_set['beta_expr']})"
        
        # Evaluate gamma schedule
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
            gamma_label = f"Gamma (prototype={param_set['gamma_prototype']}, factor={param_set['gamma_factor']})"
        elif 'gamma_based_on' in param_set and param_set['gamma_based_on'] == 'beta':
            gamma_schedule = [param_set['gamma_factor'] * b for b in beta_schedule]
            gamma_label = f"Gamma (based on beta, factor={param_set['gamma_factor']})"
        else:
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            gamma_label = f"Gamma ({param_set['gamma_expr']})"
        
        # Plot schedules
        ax.plot(time_axis, beta_schedule, label=beta_label)
        ax.plot(time_axis, gamma_schedule, label=gamma_label)
        ax.text(0.02, 0.95, f"Alpha: {param_set['alpha']}", transform=ax.transAxes, verticalalignment='top')
        ax.set_title(f"SIGMOID Model")
        
    elif model_type == ModelType.FIXED_AMPLITUDE:
        # Evaluate gamma schedule
        if 'gamma_is_prototype' in param_set and param_set['gamma_is_prototype']:
            prototype_schedule = compile_schedule(param_set['gamma_prototype'])(num_steps, env)
            gamma_schedule = [param_set['gamma_factor'] * p for p in prototype_schedule]
            gamma_label = f"Gamma (prototype={param_set['gamma_prototype']}, factor={param_set['gamma_factor']})"
        else:
            gamma_schedule = compile_schedule(param_set['gamma_expr'])(num_steps, env)
            gamma_label = f"Gamma ({param_set['gamma_expr']})"
        
        # Plot schedule
        ax.plot(time_axis, gamma_schedule, label=gamma_label)
        ax.set_title(f"FIXED_AMPLITUDE Model")
    
    elif model_type == ModelType.CIM:
        # Evaluate beta schedule
        if 'beta_is_prototype' in param_set and param_set['beta_is_prototype']:
            prototype_schedule = compile_schedule(param_set['beta_prototype'])(num_steps, env)
            beta_schedule = [param_set['beta_factor'] * p for p in prototype_schedule]
            beta_label = f"Beta (prototype={param_set['beta_prototype']}, factor={param_set['beta_factor']})"
        else:
            beta_schedule = compile_schedule(param_set['beta_expr'])(num_steps, env)
            beta_label = f"Beta ({param_set['beta_expr']})"
        
        # Plot beta schedule
        ax.plot(time_axis, beta_schedule, label=beta_label)
        
        # Add alpha, B, and zeta as text annotations
        alpha_text = f"Alpha: {param_set['alpha']}"
        
        # Display B information (either direct or scaled)
        if 'B_num_vertices' in param_set:
            B = param_set['B_num_vertices'] / graph['num_vertices']
            B_text = f"B: {B:.4f} (B_num_vertices: {param_set['B_num_vertices']} ÷ {graph['num_vertices']})"
        else:
            B_text = f"B: {param_set['B']}"
            
        zeta_text = f"Zeta: {param_set['zeta']}"
        
        ax.text(0.02, 0.95, alpha_text, transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.90, B_text, transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.85, zeta_text, transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(f"CIM Model")
    
    # Finalize the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Parameter Value")
    ax.legend()
    ax.grid(True)
    
    # Save the plot with sanitized filename
    sanitized_id = sanitize_filename(param_set['id'])
    sanitized_model = sanitize_filename(model_type.name)
    sanitized_graph = sanitize_filename(graph['name'])
    
    filename = f"{sanitized_model}_{sanitized_id}_{sanitized_graph}.png"
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def plot_config_schedules(config, graph_files, out_dir, config_basename):
    """Plot all schedules in a configuration by generating visualizations"""

    # Create output directory
    plot_schedule_dir = os.path.join(out_dir, f"schedule_plots_{config_basename}")
    os.makedirs(plot_schedule_dir, exist_ok=True)
    
    results = []
    
    # Use just the first graph file for visualizations to avoid redundancy
    if graph_files:
        graph = load_graph(graph_files[0])
        
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
            
            # Generate visualizations for each parameter set
            for param_set in param_sets:
                filepath = visualize_schedules(
                    graph, model_type, param_set, 
                    sim_params['T'], sim_params['dt'], 
                    plot_schedule_dir
                )
                
                results.append({
                    'graph': graph['name'],
                    'model': model_type.name,
                    'param_id': param_set['id'],
                    'visualization': filepath
                })
    
    # Create an HTML index file for easy viewing
    index_path = os.path.join(plot_schedule_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write("<html><head><title>Schedule Verification</title></head><body>\n")
        f.write("<h1>Schedule Verification Results</h1>\n")
        
        # Group by model type
        for model_name in set(r['model'] for r in results):
            f.write(f"<h2>{model_name} Model</h2>\n")
            model_results = [r for r in results if r['model'] == model_name]
            
            for result in model_results:
                rel_path = os.path.basename(result['visualization'])
                f.write(f"<div><h3>{result['param_id']}</h3>\n")
                f.write(f"<p>Graph: {result['graph']}</p>\n")
                f.write(f"<img src='{rel_path}' style='max-width:800px;'><br>\n")
                f.write("</div>\n")
        
        f.write("</body></html>")
    
    print(f"Schedule verification complete. {len(results)} visualizations generated.")
    print(f"View results at: {index_path}")
    
    return index_path

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
        parser.add_argument('--estimate_wall_time', nargs='?', const=True, type=int, metavar='RANKS',
                           help='Only estimate wall time without running the sweep; optionally specify number of ranks')
        parser.add_argument('--plot_schedules', action='store_true',
                           help='Generate visualizations of schedules without running simulations')
        parser.add_argument('--dry_run', action='store_true',
                           help='Preview hyperparameter loading without running simulations')
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
        assumed_time_per_step_us = config.get('assumed_time_per_step_us', 100)
        
        # Make sure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Identify graph files
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

        # Load hyperparam table if specified
        hyperparam_table = None
        if 'hyperparam_table' in config and config['hyperparam_table']:
            hyperparam_table_path = config['hyperparam_table']
            # Make path absolute if it's relative
            if not os.path.isabs(hyperparam_table_path):
                # Resolve relative to the directory containing the config file
                config_dir = os.path.dirname(os.path.abspath(args.config))
                hyperparam_table_path = os.path.join(config_dir, hyperparam_table_path)

            try:
                hyperparam_table = load_hyperparams_from_csv(hyperparam_table_path)
            except Exception as e:
                print(f"Error loading hyperparam table: {e}")
                if use_mpi:
                    comm.Abort(1)
                else:
                    sys.exit(1)

        # If dry-run is requested, show preview and exit
        if args.dry_run:
            print_hyperparam_summary(config, graph_files, hyperparam_table)
            if use_mpi:
                # Signal all ranks to exit
                tasks = []  # Empty task list signals exit
                out_dir = None
            else:
                sys.exit(0)

        # If schedule plot is requested, do that and exit
        if args.plot_schedules:
            plot_config_schedules(config, graph_files, out_dir, config_basename)
            # Exit without running simulations
            if use_mpi:
                # Signal all ranks to exit
                tasks = []  # Empty task list signals exit
                out_dir = None
            else:
                sys.exit(0)
        
        # Build task list
        tasks = []
        for graph_file in graph_files:
            graph_name = Path(graph_file).stem

            # Process each model type
            for model_name, model_params in config.get('models', {}).items():
                try:
                    model_type = ModelType[model_name]
                except KeyError:
                    print(f"Warning: Unknown model type {model_name}, skipping")
                    continue

                # Get model-specific simulation parameters
                sim_params = get_model_specific_params(config, model_name)

                # Expand T and dt to support sweeps
                T_raw = sim_params['T']
                if isinstance(T_raw, list):
                    T_values = expand_param_values(T_raw)
                elif isinstance(T_raw, str):
                    T_values = expand_param_values([T_raw])
                else:
                    T_values = [float(T_raw)]

                dt_raw = sim_params['dt']
                if isinstance(dt_raw, list):
                    dt_values = expand_param_values(dt_raw)
                elif isinstance(dt_raw, str):
                    dt_values = expand_param_values([dt_raw])
                else:
                    dt_values = [float(dt_raw)]

                num_states = int(sim_params['num_states']) if not isinstance(sim_params['num_states'], list) else int(sim_params['num_states'][0])
                noise_factor = float(sim_params['noise_factor']) if not isinstance(sim_params['noise_factor'], list) else float(sim_params['noise_factor'][0])

                # Check if we should use hyperparams from the hyperparam table
                hyperparams_from_csv = None
                if hyperparam_table and (model_name, graph_name) in hyperparam_table:
                    hyperparams_from_csv = hyperparam_table[(model_name, graph_name)]
                    print(f"Using hyperparams from table for {model_name}/{graph_name}: {hyperparams_from_csv['id']}")

                # Generate tasks for each (T, dt) combination
                for T in T_values:
                    for dt in dt_values:
                        # Generate all parameter sets for this model with these T, dt
                        param_sets = generate_param_sets(model_type, model_params, T, dt, hyperparams_from_csv)

                        # Add T, dt, num_steps to each param_set for tracking
                        for param_set in param_sets:
                            param_set['T'] = T
                            param_set['dt'] = dt
                            param_set['num_steps'] = int(np.floor(T / dt))

                            # For each parameter set, add tasks for all seeds
                            for seed in range(num_runs):
                                tasks.append({
                                    'graph_path': graph_file,
                                    'model_type': model_type,
                                    'param_set': param_set,
                                    'seed': seed,
                                    'T': T,
                                    'dt': dt,
                                    'num_states': num_states,
                                    'noise_factor': noise_factor
                                })
        
        print(f"Generated {len(tasks)} tasks")
        
        # Estimate total wall time, broken down by model type
        model_steps = {}
        model_tasks = {}
        total_steps = 0
        
        for task in tasks:
            model_name = task['model_type'].name
            steps = int(np.floor(task['T'] / task['dt']))
            
            # Track steps by model type
            if model_name not in model_steps:
                model_steps[model_name] = 0
                model_tasks[model_name] = 0
            
            model_steps[model_name] += steps
            model_tasks[model_name] += 1
            total_steps += steps
        
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
        
        expected_runtime_us = steps_per_rank * assumed_time_per_step_us
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
        print(f"Time per step        : {assumed_time_per_step_us} µs (assumed)")
        print(f"Expected runtime     : {expected_runtime}")
        print("-"*80)
        print("BREAKDOWN BY MODEL TYPE")
        print("-"*80)
        
        # Sort models by number of steps (descending)
        for model_name, steps in sorted(model_steps.items(), key=lambda x: x[1], reverse=True):
            model_percent = (steps / total_steps) * 100 if total_steps > 0 else 0
            model_runtime = timedelta(seconds=int((steps / estimation_ranks) * assumed_time_per_step_us / 1e6))
            print(f"{model_name:16s}: {model_tasks[model_name]:,} tasks | {steps:,} steps ({model_percent:.1f}%) | ~{model_runtime}")
        
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
            'assumed_time_per_step_us': assumed_time_per_step_us,
            'estimated_runtime_seconds': expected_runtime_sec
        }
        
        run_info_file = os.path.join(out_dir, f"run_info_{config_basename}.yaml")
        with open(run_info_file, 'w') as f:
            yaml.dump(run_info, f)
        
        broadcast_data = {
            'graph_files': graph_files,
            'config': config,
            'num_runs': num_runs,
            'num_tasks': len(tasks),
            'hyperparam_table': hyperparam_table
        }
    else:
        broadcast_data = None
        out_dir = None
        config_basename = None
        assumed_time_per_step_us = None

    # Broadcast task metadata
    broadcast_data = comm.bcast(broadcast_data, root=0)
    out_dir = comm.bcast(out_dir, root=0)
    config_basename = comm.bcast(config_basename, root=0)
    assumed_time_per_step_us = comm.bcast(assumed_time_per_step_us, root=0)
    
    # If no tasks, exit
    if broadcast_data['num_tasks'] == 0:
        return
    
    # Extract data
    graph_files = broadcast_data['graph_files']
    config = broadcast_data['config']
    num_runs = broadcast_data['num_runs']
    hyperparam_table = broadcast_data['hyperparam_table']

    # Each rank computes only its own tasks using deterministic indexing
    computation_start_time = time.time()

    my_tasks = []
    global_idx = 0

    # Cache for param_sets to avoid recomputing for each graph-model combination
    param_sets_cache = {}

    for graph_file in graph_files:
        graph_name = Path(graph_file).stem

        for model_name, model_params in config.get('models', {}).items():
            try:
                model_type = ModelType[model_name]
            except KeyError:
                continue

            # Get or compute param_sets for this model-graph combination
            cache_key = (model_name, graph_name)
            if cache_key not in param_sets_cache:
                sim_params = get_model_specific_params(config, model_name)

                # Expand T and dt to support sweeps
                T_raw = sim_params['T']
                if isinstance(T_raw, list):
                    T_values = expand_param_values(T_raw)
                elif isinstance(T_raw, str):
                    T_values = expand_param_values([T_raw])
                else:
                    T_values = [float(T_raw)]

                dt_raw = sim_params['dt']
                if isinstance(dt_raw, list):
                    dt_values = expand_param_values(dt_raw)
                elif isinstance(dt_raw, str):
                    dt_values = expand_param_values([dt_raw])
                else:
                    dt_values = [float(dt_raw)]

                num_states = int(sim_params['num_states']) if not isinstance(sim_params['num_states'], list) else int(sim_params['num_states'][0])
                noise_factor = float(sim_params['noise_factor']) if not isinstance(sim_params['noise_factor'], list) else float(sim_params['noise_factor'][0])

                # Check if we should use hyperparams from the hyperparam table
                hyperparams_from_csv = None
                if hyperparam_table and (model_name, graph_name) in hyperparam_table:
                    hyperparams_from_csv = hyperparam_table[(model_name, graph_name)]

                # Generate param_sets for all (T, dt) combinations
                all_param_sets = []
                for T in T_values:
                    for dt in dt_values:
                        param_sets = generate_param_sets(model_type, model_params, T, dt, hyperparams_from_csv)
                        # Add T, dt, num_steps to each param_set
                        for param_set in param_sets:
                            param_set['T'] = T
                            param_set['dt'] = dt
                            param_set['num_steps'] = int(np.floor(T / dt))
                            all_param_sets.append(param_set)

                param_sets_cache[cache_key] = {
                    'param_sets': all_param_sets,
                    'num_states': num_states,
                    'noise_factor': noise_factor,
                    'model_type': model_type
                }

            cached = param_sets_cache[cache_key]

            for param_set in cached['param_sets']:
                for seed in range(num_runs):
                    if global_idx % size == rank:
                        my_tasks.append({
                            'graph_path': graph_file,
                            'model_type': cached['model_type'],
                            'param_set': param_set,
                            'seed': seed,
                            'T': param_set['T'],
                            'dt': param_set['dt'],
                            'num_states': cached['num_states'],
                            'noise_factor': cached['noise_factor']
                        })
                    global_idx += 1
    
    print(f"Rank {rank}: Processing {len(my_tasks)} tasks")
    
    # Process assigned tasks
    local_results = []
    processing_start_time = time.time()
    last_print_time = processing_start_time
    last_print_task = 0
    total_steps_processed = 0
    
    for i, task in enumerate(my_tasks):
        try:
            graph = load_graph(task['graph_path'])
            task['graph'] = graph
            del task['graph_path']
            
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
            'assumed_time_per_step_us': assumed_time_per_step_us,
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
