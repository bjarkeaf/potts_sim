#!/usr/bin/env python3
"""
Plot convergence of figure of merit vs swept parameter for Potts machine simulations.

Usage:
    python plot_convergence.py --data <path> --conv_type <type> [--fom <metric>] [--graph_grouping <mode>] [--output_dir <path>] [--add_fits]
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
        choices=['simulation_time', 'time_step'],
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


def get_conv_param_column(conv_type: str) -> str:
    """Map convergence type to column name."""
    mapping = {
        'simulation_time': 'T',
        'time_step': 'dt'
    }
    return mapping[conv_type]


def get_axis_labels(conv_type: str, fom: str) -> Tuple[str, str]:
    """Get x and y axis labels."""
    x_labels = {
        'simulation_time': 'Simulation time, $T$',
        'time_step': r'Time step, $\mathit{dt}$'
    }
    y_labels = {
        'mean_gap': 'Relative optimality gap (%)',
        'success_rate': 'Success rate (%)'
    }
    return x_labels[conv_type], y_labels[fom]


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
        params: Fitted parameters [a, b, c] for decay or [a, k] for saturation
        model: 'decay' for exponential decay or 'saturation' for exponential saturation

    Returns:
        Formatted label string with LaTeX math
    """
    if model == 'saturation':
        # Saturation model: a * (1 - exp(-k*T))
        a, k = params

        # Format both parameters with 3 sig figs
        a_str = format_scientific_latex(a, sig_figs=3)
        k_str = format_scientific_latex(k, sig_figs=3)

        # Create label with upright exp()
        label = f"{curve_name}: ${a_str}(1-\\mathrm{{exp}}(-{k_str}T))$"
        return label

    else:
        # Decay model: a * exp(-b*T) + c
        a, b, c = params

        # Format all parameters with 3 sig figs
        a_str = format_scientific_latex(a, sig_figs=3)
        b_str = format_scientific_latex(b, sig_figs=3)
        c_str = format_scientific_latex(c, sig_figs=3)

        # Create label with upright exp()
        label = f"{curve_name}: ${a_str}\\mathrm{{exp}}(-{b_str}T) + {c_str}$"
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


def create_convergence_plot(
    df: pd.DataFrame,
    conv_type: str,
    fom: str,
    grouping: str,
    model: str,
    output_dir: Path,
    add_fits: bool = False,
    cut_outliers: bool = False
):
    """Create and save convergence plot for a specific model."""

    # Filter data for this model
    model_df = df[df['model_type'] == model].copy()

    # Calculate relative gap
    model_df['rel_gap'] = calculate_relative_gap(model_df)

    # Get convergence parameter column and values
    conv_col = get_conv_param_column(conv_type)
    conv_values = sorted(model_df[conv_col].unique())

    print(f"Processing {model} model...")
    print(f"  Swept parameter: {conv_col} in {conv_values}")

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
    ncols = min(3, n_subplots)
    nrows = (n_subplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()

    # Get axis labels
    xlabel, ylabel = get_axis_labels(conv_type, fom)

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
                conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                if len(conv_df) > 0:
                    plot_data.append(conv_df['rel_gap'].values)
                    positions.append(conv_val)

            if plot_data:
                # Calculate appropriate widths based on axis scaling
                if conv_type == 'simulation_time' and len(positions) > 1:
                    pos_array = np.array(positions)
                    if pos_array.max() / pos_array.min() > 100:
                        # Log scale: width proportional to position
                        widths = [p * 0.4 for p in positions]
                    else:
                        # Linear scale: fixed width
                        widths = np.diff(pos_array).min() * 0.6
                else:
                    widths = 0.2

                bp = ax.boxplot(plot_data, positions=positions, widths=widths)
                # Add mean markers
                means = [np.mean(d) for d in plot_data]
                ax.plot(positions, means, 'D', color='red', markersize=4, label='Mean', zorder=10)

                # Add exponential decay fits if requested
                if add_fits and len(positions) >= 3:
                    # Extract quartiles from boxplot data
                    q1_values = [np.percentile(d, 25) for d in plot_data]
                    q3_values = [np.percentile(d, 75) for d in plot_data]

                    # Convert positions to numpy array
                    T_array = np.array(positions)
                    mean_array = np.array(means)
                    q1_array = np.array(q1_values)
                    q3_array = np.array(q3_values)

                    # Create smooth T range for plotting fits
                    if conv_type == 'simulation_time' and T_array.max() / T_array.min() > 100:
                        # Log scale
                        T_smooth = np.logspace(np.log10(T_array.min()), np.log10(T_array.max()), 200)
                    else:
                        T_smooth = np.linspace(T_array.min(), T_array.max(), 200)

                    # Fit and plot three curves with individual labels

                    # Fit mean
                    params_mean = fit_exponential_decay(T_array, mean_array)
                    if params_mean is not None:
                        fit_mean = exponential_decay(T_smooth, *params_mean)
                        label_mean = format_fit_label('Mean', params_mean, model='decay')
                        ax.plot(T_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                    # Fit Q1 (25th percentile)
                    params_q1 = fit_exponential_decay(T_array, q1_array)
                    if params_q1 is not None:
                        fit_q1 = exponential_decay(T_smooth, *params_q1)
                        label_q1 = format_fit_label('Q1', params_q1, model='decay')
                        ax.plot(T_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                    # Fit Q3 (75th percentile)
                    params_q3 = fit_exponential_decay(T_array, q3_array)
                    if params_q3 is not None:
                        fit_q3 = exponential_decay(T_smooth, *params_q3)
                        label_q3 = format_fit_label('Q3', params_q3, model='decay')
                        ax.plot(T_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)

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
                    conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                    if len(conv_df) > 0:
                        sr = calculate_success_rate(conv_df)
                        success_rates.append(sr)
                        positions.append(conv_val)

                if positions:
                    ax.plot(positions, success_rates, 'o-', linewidth=2, markersize=6, label='Success rate')
                    # For per_graph line plots, always use full range
                    ax.set_ylim(0, 105)

                    # Add exponential saturation fit if requested
                    if add_fits and len(positions) >= 2:
                        T_array = np.array(positions)
                        sr_array = np.array(success_rates)

                        # Create smooth T range for plotting fit
                        if conv_type == 'simulation_time' and T_array.max() / T_array.min() > 100:
                            T_smooth = np.logspace(np.log10(T_array.min()), np.log10(T_array.max()), 200)
                        else:
                            T_smooth = np.linspace(T_array.min(), T_array.max(), 200)

                        # Fit exponential saturation to success rate
                        params_sr = fit_exponential_saturation(T_array, sr_array)
                        if params_sr is not None:
                            fit_sr = exponential_saturation(T_smooth, *params_sr)
                            # Clip to valid success rate range [0, 100]
                            fit_sr = np.clip(fit_sr, 0, 100)
                            label_fit = format_fit_label('Fit', params_sr, model='saturation')
                            ax.plot(T_smooth, fit_sr, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_fit, zorder=5)

                    ax.legend()
            else:
                # Boxplot of success rates across graphs
                plot_data = []
                positions = []
                for conv_val in conv_values:
                    conv_df = subplot_df[subplot_df[conv_col] == conv_val]
                    if len(conv_df) > 0:
                        # Calculate success rate for each graph at this conv_val
                        graph_success_rates = []
                        for graph in conv_df['graph_path'].unique():
                            graph_df = conv_df[conv_df['graph_path'] == graph]
                            sr = calculate_success_rate(graph_df)
                            graph_success_rates.append(sr)
                        plot_data.append(graph_success_rates)
                        positions.append(conv_val)

                if plot_data:
                    # Calculate appropriate widths based on axis scaling
                    if conv_type == 'simulation_time' and len(positions) > 1:
                        pos_array = np.array(positions)
                        if pos_array.max() / pos_array.min() > 100:
                            # Log scale: width proportional to position
                            widths = [p * 0.4 for p in positions]
                        else:
                            # Linear scale: fixed width
                            widths = np.diff(pos_array).min() * 0.6
                    else:
                        widths = 0.2

                    bp = ax.boxplot(plot_data, positions=positions, widths=widths)
                    # Add mean markers
                    means = [np.mean(d) for d in plot_data]
                    ax.plot(positions, means, 'D', color='red', markersize=4, label='Mean', zorder=10)

                    # Add exponential saturation fits if requested
                    if add_fits and len(positions) >= 2:
                        # Extract quartiles from boxplot data
                        q1_values = [np.percentile(d, 25) for d in plot_data]
                        q3_values = [np.percentile(d, 75) for d in plot_data]

                        # Convert positions to numpy array
                        T_array = np.array(positions)
                        mean_array = np.array(means)
                        q1_array = np.array(q1_values)
                        q3_array = np.array(q3_values)

                        # Create smooth T range for plotting fits
                        if conv_type == 'simulation_time' and T_array.max() / T_array.min() > 100:
                            # Log scale
                            T_smooth = np.logspace(np.log10(T_array.min()), np.log10(T_array.max()), 200)
                        else:
                            T_smooth = np.linspace(T_array.min(), T_array.max(), 200)

                        # Fit and plot three curves with individual labels

                        # Fit mean
                        params_mean = fit_exponential_saturation(T_array, mean_array)
                        if params_mean is not None:
                            fit_mean = exponential_saturation(T_smooth, *params_mean)
                            fit_mean = np.clip(fit_mean, 0, 100)  # Clip to valid success rate range
                            label_mean = format_fit_label('Mean', params_mean, model='saturation')
                            ax.plot(T_smooth, fit_mean, '-', color='darkgray', linewidth=1.5, alpha=0.8, label=label_mean, zorder=5)

                        # Fit Q1 (25th percentile)
                        params_q1 = fit_exponential_saturation(T_array, q1_array)
                        if params_q1 is not None:
                            fit_q1 = exponential_saturation(T_smooth, *params_q1)
                            fit_q1 = np.clip(fit_q1, 0, 100)
                            label_q1 = format_fit_label('Q1', params_q1, model='saturation')
                            ax.plot(T_smooth, fit_q1, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q1, zorder=5)

                        # Fit Q3 (75th percentile)
                        params_q3 = fit_exponential_saturation(T_array, q3_array)
                        if params_q3 is not None:
                            fit_q3 = exponential_saturation(T_smooth, *params_q3)
                            fit_q3 = np.clip(fit_q3, 0, 100)
                            label_q3 = format_fit_label('Q3', params_q3, model='saturation')
                            ax.plot(T_smooth, fit_q3, '--', color='darkgray', linewidth=1.5, alpha=0.8, label=label_q3, zorder=5)

                    ax.legend()

                    # Apply outlier-based y-limits if requested
                    if cut_outliers:
                        _, ymax = calculate_ylim_from_whiskers(bp)
                        # Ensure we stay within valid success rate range, always start at 0
                        ymax = min(ymax, 105)
                        ax.set_ylim(0, ymax)

        # Styling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Log scale for x-axis if simulation time spans orders of magnitude
        if conv_type == 'simulation_time' and len(conv_values) > 0:
            if max(conv_values) / min(conv_values) > 100:
                ax.set_xscale('log')

    # Hide unused subplots
    for idx in range(len(subplot_keys), len(axes)):
        axes[idx].axis('off')

    # Add main title
    fom_name = 'Mean Relative Gap' if fom == 'mean_gap' else 'Success Rate'
    conv_name = 'Simulation Time T' if conv_type == 'simulation_time' else 'Time Step dt'
    fig.suptitle(f'{model} Model: {fom_name} vs {conv_name}', fontsize=14, y=0.995)

    plt.tight_layout()

    # Save figure
    suffix = '_cut_outliers' if cut_outliers else ''
    filename = f'convergence_{conv_type}_{fom}_{model}_{grouping}{suffix}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Generated plot: {output_path}")


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

    # Get unique models
    models = sorted(df['model_type'].unique())

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
                cut_outliers=args.cut_outliers
            )
            plot_count += 1
        except Exception as e:
            print(f"  ERROR processing {model}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {plot_count} plots.")


if __name__ == '__main__':
    main()
