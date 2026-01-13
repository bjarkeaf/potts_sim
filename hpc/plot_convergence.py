#!/usr/bin/env python3
"""
Plot convergence of figure of merit vs swept parameter for Potts machine simulations.

Usage:
    python plot_convergence.py --data <path> --conv_type <type> [--fom <metric>] [--graph_grouping <mode>] [--output_dir <path>] [--add_fits] [--figure_mode] [--simple_filenames]

Where conv_type can be:
    - simulation_time: Sweep of total simulation time T
    - time_step: Sweep of time step dt
    - annealing: Sweep of annealing parameters (auto-detected per model)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from scipy.optimize import curve_fit
import warnings
import re
import yaml

# Styling for default value highlighting
# Figure caption should mention: "Default" indicates the parameter value used to generate main results
DEFAULT_VALUE_COLOR = 'navy'
DEFAULT_VALUE_LABEL = 'Default'

# Model name aliases for pretty-printing (with italicized q in q-PDC and q-SHIL)
MODEL_ALIAS_MAP = {
    'NEC': 'NEC',
    'QPDC': r'$q$-PDC',
    'POLYNOMIAL': 'Polynomial PM',
    'SIGMOID': 'Sigmoid PM',
    'FIXED_AMPLITUDE': r'$q$-SHIL',
    'CIM': 'Sigmoid IM',
}


def resolve_config_path(parquet_path: str) -> Optional[Path]:
    """
    Resolve the config file path from a parquet results file.

    Uses the naming convention: results_<basename>.parquet -> run_info_<basename>.yaml -> config_file

    Returns:
        Path to the config file, or None if not found
    """
    parquet_path = Path(parquet_path)
    results_dir = parquet_path.parent

    # Extract basename: results_<basename>.parquet -> <basename>
    filename = parquet_path.stem
    if filename.startswith('results_'):
        basename = filename[8:]  # len('results_') = 8
    else:
        basename = filename

    # Look for run_info file
    run_info_path = results_dir / f'run_info_{basename}.yaml'
    if not run_info_path.exists():
        return None

    # Read config_file field from run_info
    try:
        with open(run_info_path, 'r') as f:
            run_info = yaml.safe_load(f)
        config_file = run_info.get('config_file')
        if config_file:
            # Config path is relative to hpc/ directory
            config_path = results_dir.parent / config_file
            if config_path.exists():
                return config_path
    except Exception:
        pass

    return None


def extract_default_values(config_path: Path, conv_type: str) -> Dict[str, float]:
    """
    Extract default values from YAML config comments for the swept parameter.

    Parses comments like "# Default: 1e-2" or "# Default 1e1" after parameter lines.

    Args:
        config_path: Path to the YAML config file
        conv_type: Convergence type ('simulation_time', 'time_step', 'annealing')

    Returns:
        Dict mapping model name to default value (as float)
    """
    defaults = {}

    try:
        with open(config_path, 'r') as f:
            content = f.read()
    except Exception:
        return defaults

    # Determine which parameter to look for based on conv_type
    if conv_type == 'simulation_time':
        param_pattern = r'^\s*T:\s*\[.*?\]\s*#.*?Default:?\s*([\d.eE+-]+)'
    elif conv_type == 'time_step':
        param_pattern = r'^\s*dt:\s*\[.*?\]\s*#.*?Default:?\s*([\d.eE+-]+)'
    elif conv_type == 'annealing':
        # For annealing, we need to look for different parameters per model
        # Pattern matches: alpha_rate, span, etc. with a list and Default comment
        param_pattern = r'^\s*(?:alpha_rate|span):\s*\[.*?\]\s*#.*?Default:?\s*([\d.eE+-]+)'
    else:
        return defaults

    # Parse the config to find model sections and their default values
    lines = content.split('\n')
    current_model = None
    in_models_section = False

    for line in lines:
        # Check if we're entering the models section
        if line.strip() == 'models:':
            in_models_section = True
            continue

        if not in_models_section:
            continue

        # Check for model name (e.g., "  NEC:", "  POLYNOMIAL:")
        model_match = re.match(r'^  ([A-Z_]+):', line)
        if model_match:
            current_model = model_match.group(1)
            continue

        if current_model is None:
            continue

        # Look for the parameter with default value
        if conv_type == 'simulation_time':
            # Match: T: [list] # ... Default: value
            match = re.search(r'^\s*T:\s*\[.*?\].*#.*?Default:?\s*([\d.eE+-]+)', line)
        elif conv_type == 'time_step':
            # Match: dt: [list] # ... Default: value
            match = re.search(r'^\s*dt:\s*\[.*?\].*#.*?Default:?\s*([\d.eE+-]+)', line)
        elif conv_type == 'annealing':
            # Match: alpha_rate: [list] # Default: value OR span: [list] # Default: value
            match = re.search(r'^\s*(?:alpha_rate|span):\s*\[.*?\].*#.*?Default:?\s*([\d.eE+-]+)', line)
        else:
            match = None

        if match:
            try:
                defaults[current_model] = float(match.group(1))
            except ValueError:
                pass

    return defaults


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Plot convergence of figure of merit vs swept parameter'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to Parquet file containing convergence sweep results'
    )
    parser.add_argument(
        '--conv_type',
        type=str,
        required=True,
        choices=['simulation_time', 'time_step', 'annealing'],
        help='Parameter that is swept (x-axis)'
    )
    parser.add_argument(
        '--fom',
        type=str,
        default='mean_gap',
        choices=['mean_gap', 'success_rate'],
        help='Figure of merit to plot (y-axis). Default: mean_gap'
    )
    parser.add_argument(
        '--graph_grouping',
        type=str,
        default='per_graph',
        choices=['per_graph', 'all_graphs', 'by_graph_size'],
        help='How to group graphs in subplots. Default: per_graph'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output plots. Default: plots/{data_name}/'
    )
    parser.add_argument(
        '--add_fits',
        action='store_true',
        help='Add exponential decay fits to mean_gap boxplots'
    )
    parser.add_argument(
        '--cut_outliers',
        action='store_true',
        help='Set y-axis limits based on non-outlier data only, preventing extreme outliers from compressing the main data'
    )
    parser.add_argument(
        '--figure_mode',
        action='store_true',
        help='Output publication-ready figures: removes top title and saves as PDF'
    )
    parser.add_argument(
        '--simple_filenames',
        action='store_true',
        help='Use simple filenames (e.g., CIM.pdf) instead of descriptive ones'
    )
    parser.add_argument(
        '--overlay_graphs',
        action='store_true',
        help='Overlay graphs with different colors in same subplot; use model subplots instead of separate files'
    )
    return parser.parse_args()


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load and validate Parquet file."""
    print(f"Loading data from: {parquet_path}")

    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Data file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Check for required columns (adapt to actual column names)
    required_cols = ['cut_value', 'opt_cut', 'T', 'dt']
    # Use flexible column names
    if 'graph' in df.columns:
        df['graph_path'] = df['graph']  # Normalize column name
    if 'model' in df.columns:
        df['model_type'] = df['model']  # Normalize column name

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"  Shape: {df.shape}")
    print(f"  Models found: {', '.join(df['model_type'].unique())}")

    return df


def calculate_relative_gap(df: pd.DataFrame) -> pd.Series:
    """Calculate relative optimality gap: (opt_cut - cut_value) / opt_cut * 100."""
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = (df['opt_cut'] - df['cut_value']) / df['opt_cut'] * 100
    return gap


def calculate_success_rate(df_group: pd.DataFrame) -> float:
    """Calculate success rate: percentage of runs that found optimal cut."""
    return (df_group['cut_value'] == df_group['opt_cut']).sum() / len(df_group) * 100


def get_graph_sizes(df: pd.DataFrame) -> Dict[str, int]:
    """Extract number of vertices for each graph."""
    if 'num_vertices' in df.columns:
        return df.groupby('graph_path')['num_vertices'].first().to_dict()
    else:
        # If not available, return empty dict
        return {}


def parse_schedule_span(schedule_str: str) -> float:
    """
    Extract the span value from a schedule string like 'linspan(0,0.01)'.
    Returns the second number (the end value).
    """
    import re
    # Match patterns like 'linspan(start,end)' or 'lin(start,end)'
    match = re.search(r'lin(?:span)?\s*\(\s*[^,]+,\s*([^)]+)\s*\)', str(schedule_str))
    if match:
        return float(match.group(1))
    # If it's already a number, return it
    try:
        return float(schedule_str)
    except (ValueError, TypeError):
        raise ValueError(f"Could not parse schedule span from: {schedule_str}")


def get_conv_param_column(conv_type: str, model_df: pd.DataFrame = None, model: str = None) -> str:
    """
    Map convergence type to column name.

    For 'annealing' type, auto-detects which annealing parameter is swept based on the model.
    """
    if conv_type in ['simulation_time', 'time_step']:
        mapping = {
            'simulation_time': 'T',
            'time_step': 'dt'
        }
        return mapping[conv_type]

    elif conv_type == 'annealing':
        # Auto-detect which annealing parameter is being swept for this model
        if model_df is None:
            raise ValueError("model_df required for annealing convergence type")

        # Define potential annealing parameters for each model
        # Order matters: first match is used
        annealing_params_by_model = {
            'NEC': ['alpha_rate'],
            'FIXED_AMPLITUDE': ['gamma_schedule', 'gamma_span'],
            'POLYNOMIAL': ['beta_schedule', 'beta_span'],
            'SIGMOID': ['beta_schedule', 'beta_span'],
            'CIM': ['beta_schedule', 'beta_span'],
        }

        # Get candidate parameters for this model
        candidates = annealing_params_by_model.get(model, [])

        # Check which candidate has multiple unique values (is being swept)
        for param in candidates:
            if param in model_df.columns:
                unique_vals = model_df[param].nunique()
                if unique_vals > 1:
                    return param

        # Fallback: search all columns for any with multiple values
        # that contain keywords like 'span', 'rate', 'schedule', 'factor'
        annealing_keywords = ['span', 'rate', 'alpha', 'beta', 'gamma', 'factor', 'schedule']
        for col in model_df.columns:
            if any(kw in col.lower() for kw in annealing_keywords):
                if model_df[col].nunique() > 1:
                    return col

        raise ValueError(f"Could not detect swept annealing parameter for model {model}")

    else:
        raise ValueError(f"Unknown convergence type: {conv_type}")


def get_axis_labels(conv_type: str, fom: str, param_name: str = None) -> Tuple[str, str]:
    """Get x and y axis labels."""
    if conv_type == 'annealing':
        # Use parameter-specific label if available
        param_labels = {
            'alpha_rate': r'Annealing rate, $\alpha_{\mathrm{rate}}$',
            'beta_span': r'Schedule span, $\beta_{\mathrm{span}}$',
            'gamma_span': r'Schedule span, $\gamma_{\mathrm{span}}$',
            'beta_schedule': r'Schedule span, $\beta_{\mathrm{span}}$',
            'gamma_schedule': r'Schedule span, $\gamma_{\mathrm{span}}$',
            'gamma_factor': r'Gamma factor, $f_\gamma$',
        }
        x_label = param_labels.get(param_name, 'Annealing parameter')
    else:
        x_labels = {
            'simulation_time': 'Simulation time, $T$',
            'time_step': r'Time step, $\mathit{dt}$'
        }
        x_label = x_labels[conv_type]

    y_labels = {
        'mean_gap': 'Relative optimality gap (%)',
        'success_rate': 'Success rate (%)'
    }
    return x_label, y_labels[fom]


def calculate_ylim_from_whiskers(boxplot_dict: dict, padding_factor: float = 0.05) -> Tuple[float, float]:
    """
    Calculate y-axis limits based on boxplot whiskers (excluding outliers).
    This ensures all whiskers, boxes, and medians are visible, but outliers
    (shown as circles) don't affect the y-axis scaling.

    Args:
        boxplot_dict: Dictionary returned by ax.boxplot()
        padding_factor: Fraction of range to add as padding (default: 0.05 = 5%)

    Returns:
        Tuple of (ymin, ymax) for axis limits
    """
    # Get all whisker endpoints (top and bottom of each boxplot)
    whiskers = boxplot_dict['whiskers']
    whisker_ends = []
    for whisker in whiskers:
        ydata = whisker.get_ydata()
        whisker_ends.extend(ydata)

    if len(whisker_ends) == 0:
        return (0, 1)  # Default fallback

    # The whisker ends define the range of non-outlier data
    data_min = min(whisker_ends)
    data_max = max(whisker_ends)
    data_range = data_max - data_min

    # Add padding to prevent whiskers from touching edges
    if data_range > 0:
        ymin = data_min - padding_factor * data_range
        ymax = data_max + padding_factor * data_range
    else:
        # Ensure minimum separation if all data is identical
        ymin = data_min - 0.1
        ymax = data_max + 0.1

    return (ymin, ymax)


def exponential_decay(T, a, b, c):
    """Exponential decay model: gap(T) = a * exp(-b * T) + c"""
    return a * np.exp(-b * T) + c


def power_law_decay(dt, a, beta, c):
    """Power-law decay model for dt convergence: gap(dt) = a * dt^beta + c"""
    return a * np.power(dt, beta) + c


def power_law_saturation(dt, SR_max, a, beta):
    """Power-law saturation model for dt convergence: SR(dt) = SR_max - a * dt^beta"""
    return SR_max - a * np.power(dt, beta)


def exponential_saturation(T, a, k):
    """Exponential saturation model: SR(T) = a * (1 - exp(-k * T))"""
    return a * (1 - np.exp(-k * T))


def format_with_sig_figs(value: float, sig_figs: int = 3) -> str:
    """
    Format a number with specified significant figures.

    Args:
        value: Number to format
        sig_figs: Number of significant figures

    Returns:
        Formatted string with exactly sig_figs significant figures
    """
    if value == 0:
        return "0"

    # Determine the order of magnitude
    exponent = int(np.floor(np.log10(abs(value))))

    # Calculate number of decimal places needed
    # For value = 0.01837 with sig_figs=3: exponent=-2, need 4 decimal places (0.0184)
    # For value = 0.774 with sig_figs=3: exponent=-1, need 2 decimal places (0.774)
    # For value = 12.5 with sig_figs=3: exponent=1, need 1 decimal place (12.5)
    dec_places = sig_figs - exponent - 1

    # Round to the appropriate number of significant figures
    rounded = round(value, dec_places)

    # Format with the calculated decimal places
    formatted = f"{rounded:.{max(0, dec_places)}f}"

    return formatted


def format_scientific_latex(value: float, sig_figs: int = 3) -> str:
    r"""
    Format a number in LaTeX scientific notation using \cdot 10^{exp} form.
    Only uses scientific notation for |exponent| >= 3, otherwise writes out the number.

    Args:
        value: Number to format
        sig_figs: Number of significant figures (default: 3)

    Returns:
        LaTeX string like "1.00 \cdot 10^{-3}" or "0.0123"
    """
    if value == 0:
        return "0"

    # Get exponent
    exponent = int(np.floor(np.log10(abs(value))))

    # Only use scientific notation if |exponent| >= 3
    if abs(exponent) >= 3:
        mantissa = value / (10 ** exponent)
        # Format mantissa with sig_figs-1 decimal places (don't strip zeros!)
        mantissa_str = f"{mantissa:.{sig_figs-1}f}"
        return f"{mantissa_str} \\cdot 10^{{{exponent}}}"
    else:
        # Write out the number normally with sig_figs
        return format_with_sig_figs(value, sig_figs)


def format_fit_label(curve_name: str, params: np.ndarray, model: str = 'decay') -> str:
    """
    Format legend label with fit parameters using 3 significant figures.

    Args:
        curve_name: Name of the curve (e.g., 'Mean', 'Q1', 'Q3', 'Fit')
        params: Fitted parameters depending on model type
        model: 'decay', 'saturation', 'power_decay', or 'power_saturation'

    Returns:
        Formatted label string with LaTeX math
    """
    if model == 'saturation':
        # Saturation model: a * (1 - exp(-k*T))
        a, k = params

        # Format both parameters with 3 sig figs
        a_str = format_scientific_latex(a, sig_figs=3)
        k_str = format_scientific_latex(k, sig_figs=3)

        # Create label with e^{...}
        label = f"{curve_name}: ${a_str}(1-e^{{-{k_str}T}})$"
        return label

    elif model == 'power_decay':
        # Power-law decay: a * dt^beta + c
        a, beta, c = params

        a_str = format_scientific_latex(a, sig_figs=3)
        beta_str = format_with_sig_figs(beta, sig_figs=2)
        c_str = format_scientific_latex(c, sig_figs=3)

        label = f"{curve_name}: ${a_str} \\cdot dt^{{{beta_str}}} + {c_str}$"
        return label

    elif model == 'power_saturation':
        # Power-law saturation: SR_max - a * dt^beta
        SR_max, a, beta = params

        SR_max_str = format_with_sig_figs(SR_max, sig_figs=3)
        a_str = format_scientific_latex(a, sig_figs=3)
        beta_str = format_with_sig_figs(beta, sig_figs=2)

        label = f"{curve_name}: ${SR_max_str} - {a_str} \\cdot dt^{{{beta_str}}}$"
        return label

    else:
        # Decay model: a * exp(-b*T) + c
        a, b, c = params

        # Format all parameters with 3 sig figs
        a_str = format_scientific_latex(a, sig_figs=3)
        b_str = format_scientific_latex(b, sig_figs=3)
        c_str = format_scientific_latex(c, sig_figs=3)

        # Create label with e^{...}
        label = f"{curve_name}: ${a_str}e^{{-{b_str}T}} + {c_str}$"
        return label


def fit_exponential_decay(T_values: np.ndarray, y_values: np.ndarray, exclude_upturn: bool = True) -> Optional[np.ndarray]:
    """
    Fit exponential decay curve to convergence data.

    Args:
        T_values: Array of T (or dt) values
        y_values: Array of gap values (mean, Q1, or Q3)
        exclude_upturn: If True, only fit to lower 75% of T range

    Returns:
        Fitted parameters [a, b, c] or None if fitting fails
    """
    if len(T_values) < 3:
        # Need at least 3 points to fit 3 parameters
        return None

    # Optionally exclude high-T points to avoid upturn
    if exclude_upturn and len(T_values) > 3:
        threshold = np.percentile(T_values, 75)
        mask = T_values <= threshold
        if mask.sum() >= 3:
            T_fit = T_values[mask]
            y_fit = y_values[mask]
        else:
            T_fit = T_values
            y_fit = y_values
    else:
        T_fit = T_values
        y_fit = y_values

    # Initial parameter guesses
    a_init = np.max(y_fit) - np.min(y_fit)
    b_init = 0.001
    c_init = np.min(y_fit)
    p0 = [a_init, b_init, c_init]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(exponential_decay, T_fit, y_fit, p0=p0, maxfev=5000)
        return popt
    except Exception as e:
        # Fit failed, silently skip
        return None


def fit_exponential_saturation(T_values: np.ndarray, y_values: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit exponential saturation curve to success rate data.

    Args:
        T_values: Array of T (or dt) values
        y_values: Array of success rate values (mean, Q1, or Q3)

    Returns:
        Fitted parameters [a, k] or None if fitting fails
    """
    if len(T_values) < 2:
        # Need at least 2 points to fit 2 parameters
        return None

    # Initial parameter guesses
    a_init = np.max(y_values) * 1.1  # Slightly above max observed
    k_init = 0.001  # Reasonable rate constant
    p0 = [a_init, k_init]

    # Set bounds: a > 0, k > 0
    bounds = ([0, 0], [np.inf, np.inf])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(exponential_saturation, T_values, y_values, p0=p0, bounds=bounds, maxfev=5000)
        return popt
    except Exception as e:
        # Fit failed, silently skip
        return None


def fit_power_law_decay(dt_values: np.ndarray, y_values: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit power-law decay curve to gap vs dt data.

    Args:
        dt_values: Array of time step values
        y_values: Array of gap values (mean, Q1, or Q3)

    Returns:
        Fitted parameters [a, beta, c] or None if fitting fails
    """
    if len(dt_values) < 3:
        return None

    # Initial parameter guesses
    a_init = np.max(y_values) - np.min(y_values)
    beta_init = 1.0  # Start with linear (weak convergence)
    c_init = np.min(y_values)
    p0 = [a_init, beta_init, c_init]

    # Bounds: a > 0, beta > 0 (expect positive exponent), c >= 0
    bounds = ([0, 0, 0], [np.inf, 2.0, np.inf])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(power_law_decay, dt_values, y_values, p0=p0, bounds=bounds, maxfev=5000)
        return popt
    except Exception:
        return None


def fit_power_law_saturation(dt_values: np.ndarray, y_values: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit power-law saturation curve to success rate vs dt data.

    Args:
        dt_values: Array of time step values
        y_values: Array of success rate values (mean, Q1, or Q3)

    Returns:
        Fitted parameters [SR_max, a, beta] or None if fitting fails
    """
    if len(dt_values) < 3:
        return None

    # Initial parameter guesses
    SR_max_init = min(np.max(y_values) * 1.1, 100)  # Cap at 100%
    a_init = SR_max_init - np.min(y_values)
    beta_init = 1.0
    p0 = [SR_max_init, a_init, beta_init]

    # Bounds: SR_max in [0, 100], a > 0, beta > 0
    bounds = ([0, 0, 0], [100, np.inf, 2.0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(power_law_saturation, dt_values, y_values, p0=p0, bounds=bounds, maxfev=5000)
        return popt
    except Exception:
        return None


def create_convergence_plot(
    df: pd.DataFrame,
    conv_type: str,
    fom: str,
    grouping: str,
    model: str,
    output_dir: Path,
    add_fits: bool = False,
    cut_outliers: bool = False,
    figure_mode: bool = False,
    default_value: Optional[float] = None,
    simple_filenames: bool = False
):
    """Create and save convergence plot for a specific model.

    Args:
        default_value: If provided, highlight this value on the x-axis with a star marker
        simple_filenames: If True, use simple filenames (e.g., CIM.pdf) instead of descriptive ones
    """

    # Filter data for this model
    model_df = df[df['model_type'] == model].copy()

    # Calculate relative gap
    model_df['rel_gap'] = calculate_relative_gap(model_df)

    # Get convergence parameter column and values
    conv_col = get_conv_param_column(conv_type, model_df=model_df, model=model)

    # Check if column contains schedule strings that need parsing
    sample_val = model_df[conv_col].iloc[0]
    is_schedule_string = isinstance(sample_val, str) and 'lin' in sample_val

    if is_schedule_string:
        # Parse schedule strings to extract numeric span values
        model_df['_conv_numeric'] = model_df[conv_col].apply(parse_schedule_span)
        # Get unique pairs of (original, numeric) for mapping
        unique_pairs = model_df[[conv_col, '_conv_numeric']].drop_duplicates().sort_values('_conv_numeric')
        conv_values_map = dict(zip(unique_pairs['_conv_numeric'], unique_pairs[conv_col]))
        conv_values = sorted(conv_values_map.keys())  # Sorted numeric values
    else:
        # Use values directly
        conv_values = sorted(model_df[conv_col].unique())
        conv_values_map = {v: v for v in conv_values}

    # Determine if log scale should be used (parameter spans 2+ orders of magnitude)
    use_log_scale = (len(conv_values) > 0 and min(conv_values) > 0 and
                     max(conv_values) / min(conv_values) >= 100)

    print(f"Processing {model} model...")
    print(f"  Swept parameter: {conv_col} in {[conv_values_map[v] for v in conv_values]}")

    # Validate sweep
    if len(conv_values) < 2:
        print(f"  WARNING: Only {len(conv_values)} unique value(s) for {conv_col}. Skipping.")
        return

    # Get graph sizes if needed
    graph_sizes = get_graph_sizes(model_df)

    # Determine subplot organization
    if grouping == 'per_graph':
        graphs = sorted(model_df['graph_path'].unique())
        subplot_keys = graphs
        subplot_titles = [Path(g).stem if '/' in g or '.' in g else g for g in graphs]
    elif grouping == 'all_graphs':
        subplot_keys = ['all']
        subplot_titles = ['All graphs combined']
    elif grouping == 'by_graph_size':
        if not graph_sizes:
            print(f"  WARNING: Graph sizes not available. Falling back to per_graph.")
            graphs = sorted(model_df['graph_path'].unique())
            subplot_keys = graphs
            subplot_titles = [Path(g).stem if '/' in g or '.' in g else g for g in graphs]
        else:
            # Group by size
            size_to_graphs = {}
            for g, size in graph_sizes.items():
                if size not in size_to_graphs:
                    size_to_graphs[size] = []
                size_to_graphs[size].append(g)
            subplot_keys = sorted(size_to_graphs.keys())
            subplot_titles = [f'Graph size: {size} vertices' for size in subplot_keys]

    print(f"  {len(subplot_keys)} subplots")

    # Create figure with subplots
    n_subplots = len(subplot_keys)
    ncols = min(2, n_subplots)
    nrows = (n_subplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.16*nrows), squeeze=False, sharey=True)
    axes = axes.flatten()

    # Get axis labels
    xlabel, ylabel = get_axis_labels(conv_type, fom, param_name=conv_col)

    # Plot each subplot
    for idx, (key, title) in enumerate(zip(subplot_keys, subplot_titles)):
        ax = axes[idx]

        # Filter data for this subplot
        if grouping == 'per_graph':
            subplot_df = model_df[model_df['graph_path'] == key]
        elif grouping == 'all_graphs':
            subplot_df = model_df
        elif grouping == 'by_graph_size':
            if graph_sizes:
                # key is size, get graphs with that size
                graphs_in_group = [g for g, s in graph_sizes.items() if s == key]
                subplot_df = model_df[model_df['graph_path'].isin(graphs_in_group)]
            else:
                subplot_df = model_df[model_df['graph_path'] == key]

        if len(subplot_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            continue

        # Plot based on FOM type
        if fom == 'mean_gap':
            # Boxplot of gap distribution
            plot_data = []
            positions = []
            for conv_val in conv_values:
                # Filter by original value or numeric value
                if is_schedule_string:
                    orig_val = conv_values_map[conv_val]
                    conv_df = subplot_df[subplot_df[conv_col] == orig_val]
                else:
                    conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                if len(conv_df) > 0:
                    plot_data.append(conv_df['rel_gap'].values)
                    positions.append(conv_val)  # Use numeric value for position

            if plot_data:
                # Calculate appropriate widths based on axis scaling
                if len(positions) > 1:
                    if use_log_scale:
                        widths = [p * 0.4 for p in positions]
                    else:
                        widths = np.diff(np.array(positions)).min() * 0.6
                else:
                    widths = 0.2

                bp = ax.boxplot(plot_data, positions=positions, widths=widths)
                # Add mean markers
                means = [np.mean(d) for d in plot_data]
                ax.plot(positions, means, 'D', color='red', markersize=4, label='Mean', zorder=10)

                # Highlight default value with vertical dashed line
                if default_value is not None:
                    matches = [i for i, p in enumerate(positions) if np.isclose(p, default_value, rtol=1e-6)]
                    if matches:
                        default_idx = matches[0]
                        ax.axvline(x=positions[default_idx], color=DEFAULT_VALUE_COLOR, linestyle='--',
                                   linewidth=1.5, alpha=0.7, label=DEFAULT_VALUE_LABEL, zorder=5)

                # Add fits if requested
                if add_fits and len(positions) >= 3:
                    # Extract quartiles from boxplot data
                    q1_values = [np.percentile(d, 25) for d in plot_data]
                    q3_values = [np.percentile(d, 75) for d in plot_data]

                    # Convert positions to numpy array
                    x_array = np.array(positions)
                    mean_array = np.array(means)
                    q1_array = np.array(q1_values)
                    q3_array = np.array(q3_values)

                    # Create smooth range for plotting fits
                    if use_log_scale:
                        x_smooth = np.logspace(np.log10(x_array.min()), np.log10(x_array.max()), 200)
                    else:
                        x_smooth = np.linspace(x_array.min(), x_array.max(), 200)

                    # Use power-law for time_step, exponential decay otherwise
                    if conv_type == 'time_step':
                        # Power-law fits for dt convergence
                        params_mean = fit_power_law_decay(x_array, mean_array)
                        if params_mean is not None:
                            fit_mean = power_law_decay(x_smooth, *params_mean)
                            label_mean = format_fit_label('Fit', params_mean, model='power_decay')
                            ax.plot(x_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                        # params_q1 = fit_power_law_decay(x_array, q1_array)
                        # if params_q1 is not None:
                        #     fit_q1 = power_law_decay(x_smooth, *params_q1)
                        #     label_q1 = format_fit_label('Q1', params_q1, model='power_decay')
                        #     ax.plot(x_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                        # params_q3 = fit_power_law_decay(x_array, q3_array)
                        # if params_q3 is not None:
                        #     fit_q3 = power_law_decay(x_smooth, *params_q3)
                        #     label_q3 = format_fit_label('Q3', params_q3, model='power_decay')
                        #     ax.plot(x_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)
                    else:
                        # Exponential decay fits for simulation_time and annealing
                        params_mean = fit_exponential_decay(x_array, mean_array)
                        if params_mean is not None:
                            fit_mean = exponential_decay(x_smooth, *params_mean)
                            label_mean = format_fit_label('Fit', params_mean, model='decay')
                            ax.plot(x_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                        # params_q1 = fit_exponential_decay(x_array, q1_array)
                        # if params_q1 is not None:
                        #     fit_q1 = exponential_decay(x_smooth, *params_q1)
                        #     label_q1 = format_fit_label('Q1', params_q1, model='decay')
                        #     ax.plot(x_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                        # params_q3 = fit_exponential_decay(x_array, q3_array)
                        # if params_q3 is not None:
                        #     fit_q3 = exponential_decay(x_smooth, *params_q3)
                        #     label_q3 = format_fit_label('Q3', params_q3, model='decay')
                        #     ax.plot(x_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)

                ax.legend()

                # Set y-limits: always start at 0 for mean_gap
                if cut_outliers:
                    _, ymax = calculate_ylim_from_whiskers(bp)
                    ax.set_ylim(0, ymax)
                else:
                    # Auto-scale upper limit but always start at 0
                    ax.set_ylim(bottom=0)

        elif fom == 'success_rate':
            if grouping == 'per_graph':
                # Line plot with markers
                success_rates = []
                positions = []
                for conv_val in conv_values:
                    # Filter by original value or numeric value
                    if is_schedule_string:
                        orig_val = conv_values_map[conv_val]
                        conv_df = subplot_df[subplot_df[conv_col] == orig_val]
                    else:
                        conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                    if len(conv_df) > 0:
                        sr = calculate_success_rate(conv_df)
                        success_rates.append(sr)
                        positions.append(conv_val)  # Use numeric value for position

                if positions:
                    ax.plot(positions, success_rates, 'o-', linewidth=2, markersize=6, label='Success rate')
                    # For per_graph line plots, always use full range
                    ax.set_ylim(0, 105)

                    # Highlight default value with vertical dashed line
                    if default_value is not None:
                        matches = [i for i, p in enumerate(positions) if np.isclose(p, default_value, rtol=1e-6)]
                        if matches:
                            default_idx = matches[0]
                            ax.axvline(x=positions[default_idx], color=DEFAULT_VALUE_COLOR, linestyle='--',
                                       linewidth=1.5, alpha=0.7, label=DEFAULT_VALUE_LABEL, zorder=5)

                    # Add fit if requested
                    min_points = 3 if conv_type == 'time_step' else 2
                    if add_fits and len(positions) >= min_points:
                        x_array = np.array(positions)
                        sr_array = np.array(success_rates)

                        # Create smooth range for plotting fit
                        if use_log_scale:
                            x_smooth = np.logspace(np.log10(x_array.min()), np.log10(x_array.max()), 200)
                        else:
                            x_smooth = np.linspace(x_array.min(), x_array.max(), 200)

                        if conv_type == 'time_step':
                            # Power-law saturation for dt convergence
                            params_sr = fit_power_law_saturation(x_array, sr_array)
                            if params_sr is not None:
                                fit_sr = power_law_saturation(x_smooth, *params_sr)
                                fit_sr = np.clip(fit_sr, 0, 100)
                                label_fit = format_fit_label('Fit', params_sr, model='power_saturation')
                                ax.plot(x_smooth, fit_sr, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_fit, zorder=5)
                        else:
                            # Exponential saturation for simulation_time and annealing
                            params_sr = fit_exponential_saturation(x_array, sr_array)
                            if params_sr is not None:
                                fit_sr = exponential_saturation(x_smooth, *params_sr)
                                fit_sr = np.clip(fit_sr, 0, 100)
                                label_fit = format_fit_label('Fit', params_sr, model='saturation')
                                ax.plot(x_smooth, fit_sr, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_fit, zorder=5)

                    ax.legend()
            else:
                # Boxplot of success rates across graphs
                plot_data = []
                positions = []
                for conv_val in conv_values:
                    # Filter by original value or numeric value
                    if is_schedule_string:
                        orig_val = conv_values_map[conv_val]
                        conv_df = subplot_df[subplot_df[conv_col] == orig_val]
                    else:
                        conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                    if len(conv_df) > 0:
                        # Calculate success rate for each graph at this conv_val
                        graph_success_rates = []
                        for graph in conv_df['graph_path'].unique():
                            graph_df = conv_df[conv_df['graph_path'] == graph]
                            sr = calculate_success_rate(graph_df)
                            graph_success_rates.append(sr)
                        plot_data.append(graph_success_rates)
                        positions.append(conv_val)  # Use numeric value for position

                if plot_data:
                    # Calculate appropriate widths based on axis scaling
                    if len(positions) > 1:
                        if use_log_scale:
                            widths = [p * 0.4 for p in positions]
                        else:
                            widths = np.diff(np.array(positions)).min() * 0.6
                    else:
                        widths = 0.2

                    bp = ax.boxplot(plot_data, positions=positions, widths=widths)
                    # Add mean markers
                    means = [np.mean(d) for d in plot_data]
                    ax.plot(positions, means, 'D', color='red', markersize=4, label='Mean', zorder=10)

                    # Highlight default value with vertical dashed line
                    if default_value is not None:
                        matches = [i for i, p in enumerate(positions) if np.isclose(p, default_value, rtol=1e-6)]
                        if matches:
                            default_idx = matches[0]
                            ax.axvline(x=positions[default_idx], color=DEFAULT_VALUE_COLOR, linestyle='--',
                                       linewidth=1.5, alpha=0.7, label=DEFAULT_VALUE_LABEL, zorder=5)

                    # Add fits if requested
                    min_points = 3 if conv_type == 'time_step' else 2
                    if add_fits and len(positions) >= min_points:
                        # Extract quartiles from boxplot data
                        q1_values = [np.percentile(d, 25) for d in plot_data]
                        q3_values = [np.percentile(d, 75) for d in plot_data]

                        # Convert positions to numpy array
                        x_array = np.array(positions)
                        mean_array = np.array(means)
                        q1_array = np.array(q1_values)
                        q3_array = np.array(q3_values)

                        # Create smooth range for plotting fits
                        if use_log_scale:
                            x_smooth = np.logspace(np.log10(x_array.min()), np.log10(x_array.max()), 200)
                        else:
                            x_smooth = np.linspace(x_array.min(), x_array.max(), 200)

                        if conv_type == 'time_step':
                            # Power-law saturation fits for dt convergence
                            params_mean = fit_power_law_saturation(x_array, mean_array)
                            if params_mean is not None:
                                fit_mean = power_law_saturation(x_smooth, *params_mean)
                                fit_mean = np.clip(fit_mean, 0, 100)
                                label_mean = format_fit_label('Fit', params_mean, model='power_saturation')
                                ax.plot(x_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                            # params_q1 = fit_power_law_saturation(x_array, q1_array)
                            # if params_q1 is not None:
                            #     fit_q1 = power_law_saturation(x_smooth, *params_q1)
                            #     fit_q1 = np.clip(fit_q1, 0, 100)
                            #     label_q1 = format_fit_label('Q1', params_q1, model='power_saturation')
                            #     ax.plot(x_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                            # params_q3 = fit_power_law_saturation(x_array, q3_array)
                            # if params_q3 is not None:
                            #     fit_q3 = power_law_saturation(x_smooth, *params_q3)
                            #     fit_q3 = np.clip(fit_q3, 0, 100)
                            #     label_q3 = format_fit_label('Q3', params_q3, model='power_saturation')
                            #     ax.plot(x_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)
                        else:
                            # Exponential saturation fits for simulation_time and annealing
                            params_mean = fit_exponential_saturation(x_array, mean_array)
                            if params_mean is not None:
                                fit_mean = exponential_saturation(x_smooth, *params_mean)
                                fit_mean = np.clip(fit_mean, 0, 100)
                                label_mean = format_fit_label('Fit', params_mean, model='saturation')
                                ax.plot(x_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                            # params_q1 = fit_exponential_saturation(x_array, q1_array)
                            # if params_q1 is not None:
                            #     fit_q1 = exponential_saturation(x_smooth, *params_q1)
                            #     fit_q1 = np.clip(fit_q1, 0, 100)
                            #     label_q1 = format_fit_label('Q1', params_q1, model='saturation')
                            #     ax.plot(x_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                            # params_q3 = fit_exponential_saturation(x_array, q3_array)
                            # if params_q3 is not None:
                            #     fit_q3 = exponential_saturation(x_smooth, *params_q3)
                            #     fit_q3 = np.clip(fit_q3, 0, 100)
                            #     label_q3 = format_fit_label('Q3', params_q3, model='saturation')
                            #     ax.plot(x_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)

                    ax.legend()

                    # For success_rate, always use 0-100% range
                    ax.set_ylim(0, 105)

        # Styling
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Only show y-axis label on left panels
        if idx % ncols == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')

        # Log scale for x-axis if parameter spans orders of magnitude
        if use_log_scale:
            ax.set_xscale('log')

    # Hide unused subplots
    for idx in range(len(subplot_keys), len(axes)):
        axes[idx].axis('off')

    # Add main title (skip in figure_mode for publication-ready output)
    if not figure_mode:
        fom_name = 'Mean Relative Gap' if fom == 'mean_gap' else 'Success Rate'
        if conv_type == 'simulation_time':
            conv_name = 'Simulation Time T'
        elif conv_type == 'time_step':
            conv_name = 'Time Step dt'
        elif conv_type == 'annealing':
            # Use the detected parameter name for title
            param_name_map = {
                'alpha_rate': 'Annealing Rate',
                'beta_span': 'Beta Schedule Span',
                'gamma_span': 'Gamma Schedule Span',
                'beta_schedule': 'Beta Schedule Span',
                'gamma_schedule': 'Gamma Schedule Span',
                'gamma_factor': 'Gamma Factor',
            }
            conv_name = param_name_map.get(conv_col, 'Annealing Parameter')
        else:
            conv_name = conv_col
        fig.suptitle(f'{model} Model: {fom_name} vs {conv_name}', fontsize=14, y=0.995)

    plt.tight_layout()

    # Save figure
    ext = 'pdf' if figure_mode else 'png'
    if simple_filenames:
        filename = f'{model}.{ext}'
    else:
        suffix = '_cut_outliers' if cut_outliers else ''
        filename = f'convergence_{conv_type}_{fom}_{model}_{grouping}{suffix}.{ext}'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Generated plot: {output_path}")


def create_overlay_plot(
    df: pd.DataFrame,
    conv_type: str,
    fom: str,
    grouping: str,
    output_dir: Path,
    add_fits: bool = False,
    cut_outliers: bool = False,
    figure_mode: bool = False,
    default_values: Optional[Dict[str, float]] = None,
    simple_filenames: bool = False
):
    """Create convergence plot with model subplots and overlaid graph series.

    Args:
        df: DataFrame with all results
        conv_type: Type of convergence ('simulation_time', 'time_step', 'annealing')
        fom: Figure of merit ('mean_gap' or 'success_rate')
        grouping: How to group data into series ('per_graph', 'by_graph_size', 'all_graphs')
        output_dir: Directory to save plots
        add_fits: Whether to add fit curves
        cut_outliers: Whether to limit y-axis based on non-outlier data
        figure_mode: Whether to produce publication-ready output (PDF, no title)
        default_values: Dict mapping model name to default parameter value
        simple_filenames: Use simple filenames
    """
    if default_values is None:
        default_values = {}

    # Calculate relative gap
    df = df.copy()
    df['rel_gap'] = calculate_relative_gap(df)

    # Get unique models
    models = sorted(df['model_type'].unique())
    print(f"Creating overlay plot with {len(models)} model subplots...")

    # Create figure with model subplots
    n_models = len(models)
    ncols = min(2, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.32*ncols, 2.88*nrows), squeeze=False)
    axes = axes.flatten()

    # Determine series based on grouping
    graph_sizes = get_graph_sizes(df)

    if grouping == 'per_graph':
        graphs = sorted(df['graph_path'].unique())
        series_keys = graphs
        series_labels = [Path(g).stem if '/' in g or '.' in g else g for g in graphs]
    elif grouping == 'by_graph_size':
        if graph_sizes:
            sizes = sorted(set(graph_sizes.values()))
            series_keys = sizes
            series_labels = [f"{s} vertices" for s in sizes]
        else:
            print("  WARNING: Graph sizes not available. Falling back to per_graph.")
            graphs = sorted(df['graph_path'].unique())
            series_keys = graphs
            series_labels = [Path(g).stem if '/' in g or '.' in g else g for g in graphs]
            grouping = 'per_graph'
    else:  # all_graphs
        series_keys = ['all']
        series_labels = ['All graphs']

    # Color palette for series
    colors = plt.cm.tab10.colors[:len(series_keys)]

    # Plot each model subplot
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        model_df = df[df['model_type'] == model]

        # Get convergence parameter for this model
        try:
            conv_col = get_conv_param_column(conv_type, model_df=model_df, model=model)
        except ValueError as e:
            print(f"  Skipping {model}: {e}")
            ax.text(0.5, 0.5, f'No data for {model}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_ALIAS_MAP.get(model, model))
            continue

        # Parse schedule strings if needed
        sample_val = model_df[conv_col].iloc[0]
        is_schedule_string = isinstance(sample_val, str) and 'lin' in sample_val

        if is_schedule_string:
            model_df = model_df.copy()
            model_df['_conv_numeric'] = model_df[conv_col].apply(parse_schedule_span)
            conv_values = sorted(model_df['_conv_numeric'].unique())
        else:
            conv_values = sorted(model_df[conv_col].unique())

        if len(conv_values) < 2:
            ax.text(0.5, 0.5, f'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_ALIAS_MAP.get(model, model))
            continue

        # Determine log scale
        use_log_scale = (min(conv_values) > 0 and max(conv_values) / min(conv_values) >= 100)

        # Get axis labels
        xlabel, ylabel = get_axis_labels(conv_type, fom, param_name=conv_col)

        # Plot each series
        legend_handles = []
        legend_labels = []

        for series_idx, (series_key, series_label, color) in enumerate(zip(series_keys, series_labels, colors)):
            # Filter data for this series
            if grouping == 'per_graph':
                series_df = model_df[model_df['graph_path'] == series_key]
            elif grouping == 'by_graph_size':
                graphs_in_size = [g for g, s in graph_sizes.items() if s == series_key]
                series_df = model_df[model_df['graph_path'].isin(graphs_in_size)]
            else:  # all_graphs
                series_df = model_df

            if len(series_df) == 0:
                continue

            # Compute statistics per conv_value
            positions = []
            means = []
            stds = []

            for conv_val in conv_values:
                if is_schedule_string:
                    val_df = series_df[series_df['_conv_numeric'] == conv_val]
                else:
                    val_df = series_df[series_df[conv_col] == conv_val]

                if len(val_df) == 0:
                    continue

                positions.append(conv_val)

                if fom == 'mean_gap':
                    means.append(val_df['rel_gap'].mean())
                    stds.append(val_df['rel_gap'].std())
                else:  # success_rate
                    sr = calculate_success_rate(val_df)
                    means.append(sr)
                    # For success rate, compute std across runs
                    n_runs = len(val_df)
                    # Bernoulli std: sqrt(p*(1-p)/n) scaled to percentage
                    p = sr / 100
                    stds.append(np.sqrt(p * (1 - p) / n_runs) * 100 if n_runs > 0 else 0)

            if len(positions) == 0:
                continue

            positions = np.array(positions)
            means = np.array(means)
            stds = np.array(stds)

            # Plot mean with error bars (connect with lines if no fits, scatter only if fits)
            fmt = 'o' if add_fits else 'o-'
            line = ax.errorbar(positions, means, yerr=stds, fmt=fmt, color=color,
                               capsize=3, capthick=1, markersize=5, linewidth=1.5)
            legend_handles.append(line)
            legend_labels.append(series_label)

            # Add fit if requested
            if add_fits and len(positions) >= 3:
                if use_log_scale:
                    x_smooth = np.logspace(np.log10(positions.min()), np.log10(positions.max()), 200)
                else:
                    x_smooth = np.linspace(positions.min(), positions.max(), 200)

                if fom == 'mean_gap':
                    if conv_type == 'time_step':
                        params = fit_power_law_decay(positions, means)
                        if params is not None:
                            fit_y = power_law_decay(x_smooth, *params)
                            ax.plot(x_smooth, fit_y, '-', color=color, linewidth=1, alpha=0.5)
                    else:
                        params = fit_exponential_decay(positions, means)
                        if params is not None:
                            fit_y = exponential_decay(x_smooth, *params)
                            ax.plot(x_smooth, fit_y, '-', color=color, linewidth=1, alpha=0.5)
                else:  # success_rate
                    if conv_type == 'time_step':
                        params = fit_power_law_saturation(positions, means)
                        if params is not None:
                            fit_y = np.clip(power_law_saturation(x_smooth, *params), 0, 100)
                            ax.plot(x_smooth, fit_y, '-', color=color, linewidth=1, alpha=0.5)
                    else:
                        params = fit_exponential_saturation(positions, means)
                        if params is not None:
                            fit_y = np.clip(exponential_saturation(x_smooth, *params), 0, 100)
                            ax.plot(x_smooth, fit_y, '-', color=color, linewidth=1, alpha=0.5)

        # Add default value line if available
        default_value = default_values.get(model)
        if default_value is not None:
            vline = ax.axvline(x=default_value, color=DEFAULT_VALUE_COLOR, linestyle='--',
                               linewidth=1.5, alpha=0.7, zorder=5)
            legend_handles.append(vline)
            legend_labels.append(DEFAULT_VALUE_LABEL)

        # Configure axes
        ax.set_xlabel(xlabel)
        ax.set_title(MODEL_ALIAS_MAP.get(model, model))
        ax.grid(True, alpha=0.3, linestyle='--')

        if model_idx % ncols == 0:
            ax.set_ylabel(ylabel)

        if use_log_scale:
            ax.set_xscale('log')

        # Set y-limits
        if fom == 'mean_gap':
            ax.set_ylim(bottom=0)
        else:  # success_rate
            ax.set_ylim(0, 105)

        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='best', fontsize='small')

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    # Add main title (skip in figure_mode)
    if not figure_mode:
        fom_name = 'Mean Relative Gap' if fom == 'mean_gap' else 'Success Rate'
        if conv_type == 'simulation_time':
            conv_name = 'Simulation Time T'
        elif conv_type == 'time_step':
            conv_name = 'Time Step dt'
        elif conv_type == 'annealing':
            conv_name = 'Annealing Parameter'
        else:
            conv_name = conv_type
        fig.suptitle(f'{fom_name} vs {conv_name} (Overlay)', fontsize=14, y=0.995)

    plt.tight_layout()

    # Save figure
    ext = 'pdf' if figure_mode else 'png'
    if simple_filenames:
        filename = f'{conv_type}.{ext}'
    else:
        suffix = '_cut_outliers' if cut_outliers else ''
        filename = f'convergence_{conv_type}_{fom}_overlay_{grouping}{suffix}.{ext}'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated overlay plot: {output_path}")


def main():
    """Main execution function."""
    args = parse_args()

    # Load data
    try:
        df = load_data(args.data)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        data_name = Path(args.data).stem
        # Strip 'results_' prefix if present
        if data_name.startswith('results_'):
            data_name = data_name[8:]  # len('results_') = 8
        output_dir = Path('plots') / data_name

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Resolve config and extract default values
    config_path = resolve_config_path(args.data)
    default_values = {}
    if config_path:
        print(f"Config file: {config_path}")
        default_values = extract_default_values(config_path, args.conv_type)
        if default_values:
            print(f"Default values: {default_values}")
    else:
        print("Config file: not found (default highlighting disabled)")

    # Get unique models
    models = sorted(df['model_type'].unique())

    if args.overlay_graphs:
        # Create single overlay plot with model subplots
        try:
            create_overlay_plot(
                df=df,
                conv_type=args.conv_type,
                fom=args.fom,
                grouping=args.graph_grouping,
                output_dir=output_dir,
                add_fits=args.add_fits,
                cut_outliers=args.cut_outliers,
                figure_mode=args.figure_mode,
                default_values=default_values,
                simple_filenames=args.simple_filenames
            )
            plot_count = 1
        except Exception as e:
            print(f"ERROR creating overlay plot: {e}")
            import traceback
            traceback.print_exc()
            plot_count = 0
    else:
        # Create plots for each model
        plot_count = 0
        for model in models:
            try:
                create_convergence_plot(
                    df=df,
                    conv_type=args.conv_type,
                    fom=args.fom,
                    grouping=args.graph_grouping,
                    model=model,
                    output_dir=output_dir,
                    add_fits=args.add_fits,
                    cut_outliers=args.cut_outliers,
                    figure_mode=args.figure_mode,
                    default_value=default_values.get(model),
                    simple_filenames=args.simple_filenames
                )
                plot_count += 1
            except Exception as e:
                print(f"  ERROR processing {model}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. Generated {plot_count} plots.")


if __name__ == '__main__':
    main()
