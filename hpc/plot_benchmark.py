#!/usr/bin/env python3
"""
Plot benchmark results from Potts model parameter sweep.

Example usage:
    # Basic usage - plots will be saved to plots/<series_name>/
    python plot_benchmark.py --results results_max_3_cut_G70.parquet
    
    # Specify custom output directory
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --out_dir my_plots
    
    # Control model order in plots
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --model_order "NEC,QPDC,POLYNOMIAL,SIGMOID"
    
    # Generate publication-ready PDFs without titles
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --figure_mode
    
    # Generate statistics table (CSV) for best hyperparameters
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --table
    
    # Use minimum gap instead of mean gap for selecting best hyperparameters
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --best_hyperparams min_gap
    
    # Combine multiple options
    python plot_benchmark.py --results results_max_3_cut_G70.parquet --figure_mode --table --best_hyperparams mean_gap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import argparse
from pathlib import Path
import os
from collections import defaultdict
import seaborn as sns
from cycler import cycler
import warnings
import re
from potts_utils import get_best_hyperparams, save_best_hyperparams_csv

SERIES_NAME = None
FILE_EXT = "png"  # Default file extension
ADD_TITLES = True  # Default is to add titles
PROBLEM_TYPE = None  # Problem type derived from series name

# CLI flags:
#   --results     Path to the results.parquet file (required)
#   --out_dir     Directory where plots will be saved (default: plots/<series_name>)
#   --model_order Comma-separated list of model names in the desired legend/order
#   --figure_mode Output plots as PDF without titles (for publications)
#   --table       Generate CSV table with statistics for best hyperparameter combinations

# Apply theme
sns_palette_name = "muted"
sns.set_theme(style="white", palette=sns_palette_name)
plt.rc('font', family='Liberation sans')
plt.rc('axes', prop_cycle=cycler('color', sns.color_palette(sns_palette_name)))
PALETTE = sns.color_palette(sns_palette_name)

def save_figure(filename_base, out_dir):
    """Save figure with proper extension based on figure mode"""
    filename = f"{filename_base}.{FILE_EXT}"
    dpi = 200 if FILE_EXT == "png" else None  # DPI only needed for raster formats
    plt.savefig(os.path.join(out_dir, filename), dpi=dpi)
    print(f"Saved plot to {os.path.join(out_dir, filename)}")

def get_model_color(model, model_order, color_order=None):
    """
    Get the color for a given model from the palette.

    Parameters:
    - model: Model name
    - model_order: List of models in visual order
    - color_order: Optional list of models in color assignment order

    Returns:
    - Color from PALETTE
    """
    if color_order is not None and model in color_order:
        color_idx = color_order.index(model)
    elif model in model_order:
        color_idx = model_order.index(model)
    else:
        # Fallback: use position 0
        color_idx = 0

    return PALETTE[color_idx % len(PALETTE)]

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from Potts model parameter sweep results')
    parser.add_argument('--plot_type', type=str, required=False, choices=['success_rate', 'optimality_gap'], help='Type of plot to generate')
    parser.add_argument('--results', type=str, required=True, help='Path to results.parquet file')
    parser.add_argument('--out_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--model_order', type=str, default=None, help='Comma-separated list of models in desired order')
    parser.add_argument('--color_order', type=str, default=None, help='Comma-separated list of models in desired color assignment order (independent of visual order)')
    parser.add_argument('--figure_mode', action='store_true', help='Output as PDF without titles (for publications)')
    parser.add_argument('--table', action='store_true', help='Generate CSV table with statistics for best hyperparameter combinations')
    parser.add_argument('--best_hyperparams', type=str, default=None, choices=['mean_gap', 'min_gap', 'success_rate'], help='Metric for selecting best hyperparameters: mean_gap (minimize mean), min_gap (minimize minimum), or success_rate (maximize success rate). Default depends on --plot_type.')
    parser.add_argument('--graph_grouping', type=str, default='by_graph_size', choices=['per_graph', 'all_graphs', 'by_graph_size'], help='How to group graphs in plots: per_graph (one subplot per graph with scatter plot), all_graphs (one subplot averaging over all graphs with box plot), by_graph_size (subplots by graph size with box plots). Default: by_graph_size')
    parser.add_argument('--plot_hyperparams', action='store_true', help='Generate hyperparameter sweep plots')
    args = parser.parse_args()
    
    # Set default best_hyperparams based on plot_type
    if args.best_hyperparams is None:
        if args.plot_type == 'success_rate':
            args.best_hyperparams = 'success_rate'
        else:
            args.best_hyperparams = 'mean_gap'

    # Set global variables for figure mode
    global FILE_EXT, ADD_TITLES
    if args.figure_mode:
        FILE_EXT = "pdf"
        ADD_TITLES = False
    
    # infer series name and adjust default output directory
    input_path = Path(args.results)
    if input_path.name.startswith('results_') and input_path.suffix == '.parquet':
        series_name = input_path.stem[len('results_'):]
    else:
        series_name = input_path.stem
    global SERIES_NAME, PROBLEM_TYPE
    SERIES_NAME = series_name
    # derive problem type from series_name, e.g. any “max-k-cut” becomes “Max-k-Cut”, otherwise default to Max-3-Cut
    name = series_name.lower()
    m = re.search(r"max-(\d+)-cut", name)
    if m:
        PROBLEM_TYPE = f"Max-{m.group(1)}-Cut"
    else:
        PROBLEM_TYPE = "Max-3-Cut"
    default_out = parser.get_default('out_dir')
    if args.out_dir == default_out:
        args.out_dir = str(Path(default_out) / series_name)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load results
    results = pd.read_parquet(args.results)
    
    # Alias mapping for certain model keys to nicer labels
    alias_map = {
        'QPDC':             'q-PDC',
        'POLYNOMIAL':      'Polynomial PM',
        'NEC':             'NEC',
        'SIGMOID':         'Sigmoid PM',
        'FIXED_AMPLITUDE': 'q-SHIL',
        'CIM':              'Reference IM'
    }
    results['model'] = results['model'].replace(alias_map)
    
    print(f"Loaded {len(results)} results")
    print(f"Models: {results['model'].unique()}")
    print(f"Graphs: {len(results['graph'].unique())} unique graphs")
    
    # Check if success rate is already calculated, if not calculate it
    if 'reached_optimum_cut' not in results.columns and 'opt_energy' in results.columns:
        results['reached_optimum_cut'] = results['energy'] == results['opt_energy']
    
    # Calculate energy optimality gap if not already present
    if 'energy_gap' not in results.columns and 'energy' in results.columns and 'opt_energy' in results.columns:
        results['energy_gap'] = results['energy'] - results['opt_energy']
    
    # Calculate cut optimality gap if not already present
    if 'cut_value' in results.columns and 'opt_cut' in results.columns:
        if 'cut_gap' not in results.columns:
            results['cut_gap'] = results['opt_cut'] - results['cut_value']
        
        results['cut_gap'] = np.abs(results['cut_gap']) # Take absolute value for positive values

        # Calculate relative gaps
        if 'rel_cut_gap' not in results.columns:
            # Avoid division by zero
            results['rel_cut_gap'] = 100 * results['cut_gap'] / results['opt_cut'].replace(0, np.nan)
    
    # Ensure relative energy gap is calculated
    if 'energy_gap' in results.columns and 'rel_energy_gap' not in results.columns:
        results['rel_energy_gap'] = results['energy_gap'] / results['opt_energy'].replace(0, np.nan) * 100
    
    raw_order = args.model_order.split(',') if args.model_order else None
    model_order = [alias_map.get(m, m) for m in raw_order] if raw_order else None

    raw_color_order = args.color_order.split(',') if args.color_order else None
    color_order = [alias_map.get(m, m) for m in raw_color_order] if raw_color_order else None

    # Generate plot based on plot_type argument
    if args.plot_type == 'success_rate':
        if 'reached_optimum_cut' in results.columns:
            plot_success_rate_with_grouping(results, args.out_dir, model_order, args.graph_grouping, args.best_hyperparams, color_order)

            # Also generate hyperparameter plots for success rate
            if args.plot_hyperparams:
                plot_hyperparams(results, args.out_dir, model_order, args.best_hyperparams, color_order)
        else:
            print("Warning: Cannot create success rate plot - required columns not found")

    else: # if args.plot_type == 'optimality_gap': or not specified
        if 'rel_cut_gap' in results.columns:
            plot_rel_gap_distributions_by_graph(results, args.out_dir, model_order, args.best_hyperparams, color_order)
        else:
            print("Warning: Cannot create relative gap distribution plots - relative cut gap not found")

        # Also generate hyperparameter plots for optimality gap
        if 'cut_gap' in results.columns:
            plot_hyperparams(results, args.out_dir, model_order, args.best_hyperparams, color_order)
        else:
            print("Warning: Cannot create hyperparameter sweep plots - cut gap not found")

    # Generate statistics table if requested
    if args.table and 'rel_cut_gap' in results.columns:
        csv_path = os.path.join(args.out_dir, f'{SERIES_NAME}_statistics.csv')
        save_best_hyperparams_csv(results, csv_path, mode=args.best_hyperparams, model_order=model_order)
    elif args.table:
        print("Warning: Cannot create statistics table - relative cut gap not found")

def sort_models_by_performance(data, metric_col, ascending=True):
    """
    Sort models by their mean performance on the given metric.
    
    Parameters:
    - data: DataFrame with results
    - metric_col: Column name of the metric to use for sorting
    - ascending: If True, sort in ascending order (lower is better)
    
    Returns:
    - List of model names sorted by performance
    """
    # Calculate mean metric value for each model
    avg_by_model = data.groupby('model')[metric_col].mean().reset_index()
    
    # Sort models by mean metric value
    sorted_models = avg_by_model.sort_values(metric_col, ascending=ascending)['model'].tolist()
    
    return sorted_models

def plot_optimality_gap(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap'):
    """Plot best mean optimality gap per model as horizontal bars."""
    # Get best parameters for each graph-model combination (minimize cut gap)
    best_params = get_best_hyperparams(results, mode=best_hyperparams_mode)
    
    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results, 
        best_params[['graph', 'model', 'param_id']], 
        on=['graph', 'model', 'param_id']
    )
    
    # compute mean cut per model
    mean_cut = filtered_results.groupby('model')['cut_gap'].mean().sort_values()
    models = mean_cut.index.tolist()
    values = mean_cut.values

    plt.figure(figsize=(4, len(models)*0.6))
    if ADD_TITLES:
        plt.title(f'{SERIES_NAME} | Absolute optimality gap')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    max_val = values.max()
    offset = max_val * 0.01
    plt.barh(models, values, color=PALETTE[:len(models)])
    for i, v in enumerate(values):
        plt.text(v + offset, i, f'{v:.2f}', va='center', ha='left')
    plt.xlabel('Mean optimality gap (cut value)')
    plt.tight_layout()
    save_figure('optimality_gaps', out_dir)
    plt.close()

def plot_relative_optimality_gap(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap'):
    """Plot best mean relative optimality gap per model as horizontal bars."""
    # Get best parameters for each graph-model combination (minimize relative cut gap)
    best_params = get_best_hyperparams(results, mode=best_hyperparams_mode)
    
    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results, 
        best_params[['graph', 'model', 'param_id']], 
        on=['graph', 'model', 'param_id']
    )
    
    # compute mean relative gap
    mean_rel_cut = filtered_results.groupby('model')['rel_cut_gap'].mean().sort_values()
    models = mean_rel_cut.index.tolist()
    values = mean_rel_cut.values

    plt.figure(figsize=(4, len(models)*0.6))
    if ADD_TITLES:
        plt.title(f'{SERIES_NAME} | Relative optimality gap')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    max_val = values.max()
    offset = max_val * 0.01
    plt.barh(models, values, color=PALETTE[:len(models)])
    for i, v in enumerate(values):
        plt.text(v + offset, i, f'{v:.2f}%', va='center', ha='left')
    plt.xlabel('Relative optimality gap (%)')
    plt.tight_layout()
    save_figure('rel_optimality_gaps', out_dir)
    plt.close()

def add_gamma_rate_over_th(data):
    """
    For QPDC model, add gamma_rate/gamma_th column.
    This is the normalized rate of gamma increase: ε_γ / γ_th

    With γ = γ₀ + ε_γ·t and gamma = factor × prototype:
    - For linspan(start, span): ε_γ = factor × span / T
    - So ε_γ/γ_th = factor × span / (T × γ_th)

    Returns a modified copy of the dataframe
    """
    gamma_th = (256/27)**(1/4)

    if 'gamma_factor' not in data.columns or 'T' not in data.columns:
        return data

    data = data.copy()

    # Check if gamma_prototype column exists to determine the span
    if 'gamma_prototype' in data.columns:
        def compute_normalized_rate(row):
            prototype = row.get('gamma_prototype', '')
            factor = row['gamma_factor']
            T = row['T']

            if pd.isna(prototype) or prototype == '':
                # Default: assume lin(0,1) equivalent, span=1
                span = 1.0
            elif 'linspan(' in str(prototype):
                # Parse linspan(start, span) - extract span value
                match = re.search(r'linspan\s*\(\s*[^,]+\s*,\s*([^)]+)\s*\)', str(prototype))
                if match:
                    span_expr = match.group(1).strip()
                    # Evaluate span expression (may contain T)
                    try:
                        span = eval(span_expr, {'T': T, 'np': np})
                    except:
                        span = 1.0
                else:
                    span = 1.0
            elif 'lin(' in str(prototype):
                # Parse lin(start, end) - span = end - start
                match = re.search(r'lin\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', str(prototype))
                if match:
                    try:
                        start = float(match.group(1).strip())
                        end = float(match.group(2).strip())
                        span = end - start
                    except:
                        span = 1.0
                else:
                    span = 1.0
            else:
                span = 1.0

            return factor * span / (T * gamma_th)

        data['gamma_rate/gamma_th'] = data.apply(compute_normalized_rate, axis=1)
    else:
        # Fallback: assume standard lin(0,1) prototype where span=1
        data['gamma_rate/gamma_th'] = data['gamma_factor'] / (data['T'] * gamma_th)

    return data

def plot_hyperparams(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap', color_order=None):
    """
    Plot the mean relative optimality gap versus hyperparameters for each model.

    For 1D hyperparameter sweeps: Line plot of gap vs parameter
    For 2D hyperparameter sweeps: Heatmap with parameters as axes
    For 3D+ hyperparameter sweeps: Skip plotting with a message
    For no hyperparameter sweep: Skip plotting with a message

    Supports modes: 'mean_gap', 'min_gap', 'success_rate'
    """
    # Try to import seaborn for better heatmaps
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Determine the metric column based on mode
    if best_hyperparams_mode == 'success_rate':
        metric_col = 'reached_optimum_cut'
        if metric_col not in results.columns:
            print("Warning: Cannot create hyperparameter plots for success_rate - reached_optimum_cut not found")
            return
    else:
        metric_col = 'rel_cut_gap'
        if metric_col not in results.columns:
            print("Warning: Cannot create hyperparameter plots - rel_cut_gap not found")
            return
    
    # Process each model type separately
    models = model_order if model_order else sorted(results['model'].unique())
    
    for model in models:
        model_data = results[results['model'] == model].copy()
        
        # Skip if no data for this model
        if model_data.empty:
            continue
        
        # For QPDC, add gamma_rate/gamma_th and replace gamma_factor with it
        if model == 'q-PDC':
            model_data = add_gamma_rate_over_th(model_data)
            
        # Identify which hyperparameters are being swept for this model
        swept_params = identify_swept_hyperparameters(model_data, model)
        
        # Get unique graphs for this model
        graphs = sorted(model_data['graph'].unique())
        if not graphs:
            continue

        # Based on number of swept parameters, create appropriate plot
        if len(swept_params) == 0:
            # No hyperparameters being swept, skip plotting
            pass
        elif len(swept_params) == 1:
            param = swept_params[0]
            print(f"Creating 1D hyperparameter plot for {model}: {param}")
            
            # Create subplots for each graph
            num_graphs = len(graphs)
            ncols = 2
            nrows = (num_graphs + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, sharex='col', sharey='row')
            
            for i, graph in enumerate(graphs):
                row, col = i // ncols, i % ncols
                ax = axes[row, col]
                graph_data = model_data[model_data['graph'] == graph]
                
                is_leftmost = (col == 0)
                is_bottom = (col == 0 and row == nrows - 1) or (col == 1 and row == nrows - 2)
                
                if is_bottom:
                    ax.tick_params(labelbottom=True)

                plot_1d_hyperparam(graph_data, param, model, ax, graph, is_leftmost, is_bottom, i, best_hyperparams_mode)
            
            # Hide unused subplots
            for i in range(num_graphs, nrows * ncols):
                row, col = i // ncols, i % ncols
                fig.delaxes(axes[row, col])

            if ADD_TITLES:
                fig.suptitle(f'{model}, {PROBLEM_TYPE}', x=0.54, ha='center')
                
            plt.tight_layout(rect=[0, 0, 1, 1], h_pad=-1)
            filename = f'{model.lower().replace(" ", "_").replace("-", "_")}_1d_hyperparam'
            save_figure(filename, out_dir)
            plt.close(fig)

        elif len(swept_params) == 2:
            param1, param2 = swept_params
            print(f"Creating 2D hyperparameter plot for {model}: {param1} vs {param2}")
            
            # Calculate global min and max for the color bar across all graphs for this model
            if best_hyperparams_mode == 'success_rate':
                all_graph_data = model_data.groupby([param1, param2])['reached_optimum_cut'].mean() * 100
                global_min = all_graph_data.min()
                global_max = all_graph_data.max()
            elif best_hyperparams_mode == 'mean_gap':
                all_graph_data = model_data.groupby([param1, param2])['rel_cut_gap'].mean()
                global_min = all_graph_data[all_graph_data > 0].min()
                global_max = all_graph_data.max()
            else: # min_gap
                all_graph_data = model_data.groupby([param1, param2])['rel_cut_gap'].min()
                global_min = all_graph_data[all_graph_data > 0].min()
                global_max = all_graph_data.max()

            # Create subplots for each graph
            num_graphs = len(graphs)
            ncols = 2
            nrows = (num_graphs + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False, sharex='col', sharey='row')

            for i, graph in enumerate(graphs):
                row, col = i // ncols, i % ncols
                ax = axes[row, col]
                graph_data = model_data[model_data['graph'] == graph]

                is_leftmost = (col == 0)
                is_bottom = (col == 0 and row == nrows - 1) or (col == 1 and row == nrows - 2)

                if is_bottom:
                    ax.tick_params(labelbottom=True)

                plot_2d_hyperparam(graph_data, param1, param2, model, ax, graph, has_seaborn, global_min, global_max, is_leftmost, is_bottom, i, best_hyperparams_mode)

            # Hide unused subplots
            for i in range(num_graphs, nrows * ncols):
                row, col = i // ncols, i % ncols
                fig.delaxes(axes[row, col])

            if ADD_TITLES:
                fig.suptitle(f'{model}, {PROBLEM_TYPE}', x=0.54, ha='center')

            # Add a single horizontal colorbar for the entire figure
            # For success rate, don't use log norm
            if best_hyperparams_mode == 'success_rate':
                norm = None
                cbar_label = 'Success Rate (SR)'
            else:
                norm = None
                if pd.notna(global_min) and global_min < global_max:
                    norm = LogNorm(vmin=global_min, vmax=global_max)
                cbar_label = 'Mean relative optimality gap (%)'
            
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            
            # Position colorbar at the bottom
            cbar_ax = fig.add_axes([0.15, 0.065, 0.7, 0.02]) # change second for bottom margin
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(cbar_label)

            # Adjust layout to make space for colorbar at the bottom
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.tight_layout(rect=[0, 0.07, 1, 1], h_pad=-1) # change second for bottom margin
            filename = f'{model.lower().replace(" ", "_").replace("-", "_")}_2d_hyperparam'
            save_figure(filename, out_dir)
            plt.close(fig)
        else:
            print(f"Skipping hyperparameter plot for {model}: More than 2 parameters being swept ({swept_params})")

def identify_swept_hyperparameters(model_data, model=None):
    """
    Identify which hyperparameters are being swept.
    Returns a list of parameter names that have multiple unique values.
    
    For QPDC model, use gamma_rate/gamma_th instead of gamma_factor if available.
    """
    # List of potential hyperparameters to check
    potential_params = [
        'poly_order', 'gamma_factor', 'beta_factor', 
        'alpha_rate', 'r_target', 'alpha',
        'B_num_vertices', 'zeta', 'gamma_rate/gamma_th'
    ]
    
    # Check which parameters have multiple unique values
    swept_params = []
    for param in potential_params:
        if param in model_data.columns and len(model_data[param].unique()) > 1:
            # For QPDC, replace gamma_factor with gamma_rate/gamma_th if both are present
            if model == 'q-PDC' and param == 'gamma_factor' and 'gamma_rate/gamma_th' in model_data.columns:
                continue
            swept_params.append(param)
    
    return swept_params

def plot_success_rate_with_grouping(results, out_dir, model_order=None, graph_grouping='by_graph_size', best_hyperparams_mode='success_rate', color_order=None):
    """
    Plot success rates with different graph grouping strategies.

    Parameters:
    - results: DataFrame with simulation results
    - out_dir: Output directory for plots
    - model_order: List of model names in desired order (or None to auto-sort)
    - graph_grouping: How to group graphs ('per_graph', 'all_graphs', 'by_graph_size')
    - best_hyperparams_mode: Metric for selecting best hyperparameters
    - color_order: Optional list of models in color assignment order
    """
    # Get best parameters for each graph-model combination
    best_params = get_best_hyperparams(results, mode=best_hyperparams_mode)

    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results,
        best_params[['graph', 'model', 'param_id']],
        on=['graph', 'model', 'param_id']
    )

    # Calculate success rates by aggregating over runs
    success_rates = filtered_results.groupby(['graph', 'model', 'param_id'])['reached_optimum_cut'].mean().reset_index()
    success_rates['success_rate'] = success_rates['reached_optimum_cut'] * 100  # Convert to percentage
    success_rates = success_rates.drop(columns=['reached_optimum_cut'])

    # Merge back to get additional columns (num_vertices, swept parameters, etc.)
    success_rates = success_rates.merge(
        filtered_results[['graph', 'model', 'param_id', 'num_vertices']].drop_duplicates(),
        on=['graph', 'model', 'param_id']
    )

    # Determine model ordering if not provided (by mean success rate, descending)
    if model_order is None:
        model_performance = success_rates.groupby('model')['success_rate'].mean().sort_values(ascending=False)
        model_order = model_performance.index.tolist()

    # Branch based on grouping mode
    if graph_grouping == 'per_graph':
        _plot_success_rate_per_graph(success_rates, filtered_results, out_dir, model_order, color_order)
    elif graph_grouping == 'all_graphs':
        _plot_success_rate_all_graphs(success_rates, out_dir, model_order, color_order)
    elif graph_grouping == 'by_graph_size':
        _plot_success_rate_by_graph_size(success_rates, out_dir, model_order, color_order)
    else:
        print(f"Warning: Unknown graph_grouping mode '{graph_grouping}'")

def _plot_success_rate_per_graph(success_rates, filtered_results, out_dir, model_order, color_order=None):
    """
    Plot success rates with one section per graph (scatter plot style).
    Shows aggregated success rate per parameter value for each model.
    """
    # Check if there are swept parameters
    graphs = sorted(success_rates['graph'].unique())
    if not graphs:
        print("Warning: No graphs found in filtered results")
        return

    # For the first graph, identify swept parameters
    first_graph_data = filtered_results[filtered_results['graph'] == graphs[0]]
    swept_params = identify_swept_hyperparameters(first_graph_data)

    if len(swept_params) == 0:
        print("Warning: No swept parameters found for 'per_graph' mode - falling back to bar chart")
        # Could implement a bar chart fallback here
        return
    elif len(swept_params) > 1:
        print(f"Warning: Multiple swept parameters found {swept_params} - using first one: {swept_params[0]}")

    param_name = swept_params[0]

    # Merge parameter values into success_rates
    param_data = filtered_results[['graph', 'model', 'param_id', param_name]].drop_duplicates()
    success_rates_with_param = success_rates.merge(param_data, on=['graph', 'model', 'param_id'])

    # Get unique parameter values (globally across all graphs)
    param_values = sorted(success_rates_with_param[param_name].unique())
    num_param_values = len(param_values)
    num_models = len(model_order)

    # Calculate positions
    positions_per_graph = num_param_values * num_models
    graph_centers = []
    separator_positions = []

    # Prepare data structures
    scatter_data = []  # List of (position, success_rate, model) tuples
    position_to_param_value = {}

    for g_idx, graph in enumerate(graphs):
        start_pos = g_idx * positions_per_graph
        graph_centers.append(start_pos + positions_per_graph/2 - 0.5)
        if g_idx > 0:
            separator_positions.append(start_pos - 0.5)

        graph_data = success_rates_with_param[success_rates_with_param['graph'] == graph]

        for p_idx, param_value in enumerate(param_values):
            for m_idx, model in enumerate(model_order):
                position = start_pos + (p_idx * num_models) + m_idx
                position_to_param_value[position] = param_value

                # Get success rate for this combination
                point_data = graph_data[
                    (graph_data['model'] == model) &
                    (graph_data[param_name] == param_value)
                ]

                if not point_data.empty:
                    sr = point_data['success_rate'].iloc[0]
                    scatter_data.append((position, sr, model))

    # Create plot
    fig_width = min(12, 0.15 * len(graphs) * positions_per_graph + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Plot scatter points
    for position, sr, model in scatter_data:
        ax.scatter(position, sr, color=get_model_color(model, model_order, color_order), s=50, alpha=0.7)

    # Add vertical separators between graphs
    for pos in separator_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)

    # Add horizontal line at 100% (perfect success)
    #ax.axhline(y=100, color='green', linestyle='-', linewidth=1, alpha=0.5)

    # Set x-axis labels
    ax.set_xticks(graph_centers)
    ax.set_xticklabels([f"{g}" for g in graphs])
    ax.set_xlabel('Graph')
    ax.set_ylabel('Success Rate (SR)')
    ax.set_ylim(-2,102)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [plt.scatter([], [], color=get_model_color(model, model_order, color_order), s=50, label=model)
                      for model in model_order]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))

    if ADD_TITLES:
        ax.set_title(f'{SERIES_NAME} | Success rate by graph (swept parameter: {param_name})')

    plt.tight_layout()
    save_figure('success_rate_per_graph', out_dir)
    plt.close()

def _plot_success_rate_all_graphs(success_rates, out_dir, model_order, color_order=None):
    """
    Plot success rates with all graphs grouped together (box plot).
    """
    # Prepare data for box plot
    boxplot_data = []

    for model in model_order:
        model_data = success_rates[success_rates['model'] == model]
        boxplot_data.append(model_data['success_rate'].tolist())

    # Create plot
    fig, ax = plt.subplots(figsize=(max(6, len(model_order) * 0.8), 5))

    # Create box plot
    bp = ax.boxplot(boxplot_data, tick_labels=model_order, patch_artist=True,
                   widths=0.6, medianprops={'color': 'black'})

    # Color boxes
    for i, box in enumerate(bp['boxes']):
        model = model_order[i]
        box.set_facecolor(get_model_color(model, model_order, color_order))

    # Add horizontal reference lines at 100% and 0%
    ax.axhline(y=100, color='gray', linestyle='-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)

    ax.set_ylabel('Success Rate (SR)')
    ax.set_ylim(-2,102)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    if ADD_TITLES:
        ax.set_title(f'{SERIES_NAME} | Success rate distribution (all graphs)')

    plt.tight_layout()
    save_figure('success_rate_all_graphs', out_dir)
    plt.close()

def _plot_success_rate_by_graph_size(success_rates, out_dir, model_order, color_order=None):
    """
    Plot success rates grouped by graph size (box plot style).
    Similar visual style to plot_rel_gap_distributions_by_graph.
    """
    # Get unique graph sizes
    sizes = sorted(success_rates['num_vertices'].unique())
    num_models = len(model_order)

    # Calculate positions
    positions_per_size = num_models
    size_centers = []
    separator_positions = []

    # Prepare data structures
    boxplot_data = {}
    position_model_map = {}

    for s_idx, size in enumerate(sizes):
        start_pos = s_idx * positions_per_size
        size_centers.append(start_pos + positions_per_size/2 - 0.5)
        if s_idx > 0:
            separator_positions.append(start_pos - 0.5)

        size_data = success_rates[success_rates['num_vertices'] == size]

        for m_idx, model in enumerate(model_order):
            position = start_pos + m_idx
            model_data = size_data[size_data['model'] == model]

            if not model_data.empty:
                boxplot_data[position] = model_data['success_rate'].tolist()
            else:
                boxplot_data[position] = []

            position_model_map[position] = model

    # Sort positions and filter out empty ones
    sorted_positions = sorted(boxplot_data.keys())
    all_data = [boxplot_data[pos] for pos in sorted_positions]
    mask = [len(data) > 0 for data in all_data]
    valid_positions = [pos for i, pos in enumerate(sorted_positions) if mask[i]]
    valid_data = [data for data in all_data if len(data) > 0]

    # Create plot
    fig_width = min(10, 0.4 * len(sizes) * num_models + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Create box plots
    bp = ax.boxplot(valid_data, positions=valid_positions, patch_artist=True,
                   widths=0.7, medianprops={'color': 'black'})

    # Color boxes
    for i, box in enumerate(bp['boxes']):
        model = position_model_map[valid_positions[i]]
        box.set_facecolor(get_model_color(model, model_order, color_order))

    # Enhance median lines for collapsed boxes
    y_min, y_max = ax.get_ylim()
    collapse_threshold = (y_max - y_min) * 0.01

    for i, box in enumerate(bp['boxes']):
        box_extent = box.get_path().get_extents()
        box_height = box_extent.height
        if box_height < collapse_threshold:
            model = position_model_map[valid_positions[i]]
            median_line = bp['medians'][i]
            median_line.set_linewidth(4)
            median_line.set_color(get_model_color(model, model_order, color_order))

    # Add vertical separators between size groups
    for pos in separator_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)

    # Add horizontal reference lines at 100% and 0%
    ax.axhline(y=100, color='gray', linestyle='-', linewidth=1, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.7)

    # Set x-axis labels as numbers only
    ax.set_xticks(size_centers)
    ax.set_xticklabels([f"{size}" for size in sizes])
    ax.set_xlabel('Number of vertices in graph')
    ax.set_ylabel('Success Rate (SR)')
    ax.set_ylim(-2,102)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=get_model_color(model, model_order, color_order),
                                    edgecolor='black') for model in model_order]

    ax.legend(legend_elements, model_order,
             loc='center left', bbox_to_anchor=(1.02, 0.5))

    if ADD_TITLES:
        ax.set_title(f'{SERIES_NAME} | Success rate by graph size')

    plt.tight_layout()
    save_figure('success_rate_by_graph_size', out_dir)
    plt.close()

def plot_rel_gap_distributions_by_graph(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap', color_order=None):
    """
    Create boxplots showing relative optimality gap distributions by graph and model.

    For each graph-model combination, only the best hyperparameter set is used
    (the one with smallest mean relative optimality gap).

    If there is a consistent large gap between model performances, a broken y-axis is used.
    """
    # Get best parameters for each graph-model combination (minimize cut gap)
    best_params = get_best_hyperparams(results, mode=best_hyperparams_mode)
    
    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results, 
        best_params[['graph', 'model', 'param_id']], 
        on=['graph', 'model', 'param_id']
    )
    
    # Determine model ordering if not provided
    if model_order is None:
        print("Using model order based on mean relative optimality gap")
        # Order models by their mean relative optimality gap across all graphs
        if best_hyperparams_mode == 'mean_gap':
            model_performance = filtered_results.groupby('model')['rel_cut_gap'].mean().sort_values()
        else: # min_gap
            # compute each graph‐model’s min rel_cut_gap
            min_by_graph = (
                filtered_results
                .groupby(['graph','model'])['rel_cut_gap']
                .min()
                .reset_index()
            )
            # now average those minima over all graphs, then sort
            model_performance = (
                min_by_graph
                .groupby('model')['rel_cut_gap']
                .mean()
                .sort_values()
            )
            model_order = model_performance.index.tolist()
        model_order = model_performance.index.tolist()

    
    # Get unique graphs and sort them alphanumerically
    graphs_df = filtered_results[['graph', 'num_vertices']].drop_duplicates()
    graphs = sorted(graphs_df['graph'].unique())  # Sort alphanumerically
    
    # Calculate the total number of positions needed
    num_models = len(model_order)
    total_width = num_models * len(graphs)
    
    # --- Check for broken axis ---
    # Calculate median rel_cut_gap for each model across all graphs
    model_medians = filtered_results.groupby('model')['rel_cut_gap'].median().sort_values()
    
    # Find the largest gap(s) between consecutive sorted medians
    # Determine breaks based on gaps between distributions (no overlap region)
    use_broken_axis = False
    break_indices = []
    gap_threshold = 0.9  # any positive gap indicates a region with no data

    # Ensure we have at least two models to compare
    if len(model_medians) > 1:
        models = model_medians.index.tolist()
        for i in range(len(models) - 1):
            lower_model = models[i]
            upper_model = models[i + 1]

            # max of lower distribution, min of upper distribution
            lower_max = filtered_results.loc[
                filtered_results['model'] == lower_model, 'rel_cut_gap'
            ].max()
            upper_min = filtered_results.loc[
                filtered_results['model'] == upper_model, 'rel_cut_gap'
            ].min()

            # if there's a positive gap, mark a break
            if pd.notna(lower_max) and pd.notna(upper_min):
                gap = upper_min - lower_max
                if gap > gap_threshold:
                    break_indices.append(i + 1)

        if break_indices:
            use_broken_axis = True

    # Prepare the plot with exact sizing
    fig_width = min(10, 0.3 * total_width + 2)  # Add margin for labels
    fig_height = 5
    
    if use_broken_axis:
        # Split models into segments based on break_indices
        split_points = [0] + break_indices + [len(model_medians)]
        model_segments = [model_medians.index[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]
        

        # For each segment, determine y-limits
        segment_limits = []
        margin = 0.015
        all_data_max = filtered_results['rel_cut_gap'].max()

        # Compute the total width of all gaps between segments
        gap_widths = []
        for idx in break_indices:
            # The gap is the difference between the upper of the lower segment and the lower of the upper segment
            lower_models = model_medians.index[:idx]
            upper_models = model_medians.index[idx:]
            lower_max = filtered_results[filtered_results['model'].isin(lower_models)]['rel_cut_gap'].max()
            upper_min = filtered_results[filtered_results['model'].isin(upper_models)]['rel_cut_gap'].min()
            if pd.notna(lower_max) and pd.notna(upper_min):
                gap_widths.append(upper_min - lower_max)
        total_gap_width = sum(gap_widths) if gap_widths else 0

        for i, models in enumerate(model_segments):
            seg_data = filtered_results[filtered_results['model'].isin(models)]
            seg_min = seg_data['rel_cut_gap'].min()
            seg_max = seg_data['rel_cut_gap'].max()
            # Add margin
            seg_min = seg_min - margin * abs(seg_min) if pd.notna(seg_min) else 0
            seg_max = seg_max + margin * abs(seg_max) if pd.notna(seg_max) else 1
            # For the lowest segment, extend min below zero
            if i == 0:
                seg_min = -0.02 * (all_data_max - total_gap_width)
            segment_limits.append((seg_min, seg_max))

        # Calculate height ratios proportional to y-range
        height_ratios = [seg_max - seg_min for seg_min, seg_max in segment_limits]

        # --- FIX: Reverse so top axis is for largest values ---
        model_segments = model_segments[::-1]
        segment_limits = segment_limits[::-1]
        height_ratios  = height_ratios[::-1]

        # Create subplots for each segment
        fig, axes = plt.subplots(len(model_segments), 1, sharex=True, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': height_ratios})

        # Set y-limits and hide spines between axes
        for i, ax in enumerate(axes):
            ax.set_ylim(*segment_limits[i])
            ax.tick_params(axis='y', which='both', left=True)
            if i < len(axes) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', which='both', bottom=False)
            if i > 0:
                ax.spines['top'].set_visible(False)

        # Set consistent y-tick formatting, 2 decimal places
        formatter = FormatStrFormatter('%.2f')
        for ax in axes:
            ax.yaxis.set_major_formatter(formatter)

        # --- Force tick step size to 0.2 for all axes ---
        tick_step = 0.2
        # Find the global min and max for ticks
        combined_min = min(seg[0] for seg in segment_limits)
        combined_max = max(seg[1] for seg in segment_limits)
        # Generate ticks at 0.2 intervals within the global range
        ticks = np.arange(np.floor(combined_min / tick_step) * tick_step,
                  np.ceil(combined_max / tick_step) * tick_step + tick_step/2,
                  tick_step)
        # Set ticks for each axis within its segment limits
        for i, ax in enumerate(axes):
            seg_min, seg_max = segment_limits[i]
            seg_ticks = [t for t in ticks if seg_min <= t <= seg_max]
            ax.set_yticks(seg_ticks)

        # Add diagonal lines to indicate breaks
        d = .015
        h_ref = max(height_ratios)
        for i in range(len(axes)-1):
            ax_top = axes[i]
            ax_bottom = axes[i+1]
            # Scale d_y inversely with panel height for consistent physical size
            d_y_top = d * h_ref / height_ratios[i]
            d_y_bottom = d * h_ref / height_ratios[i+1]
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d_y_top, +d_y_top), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d_y_top, +d_y_top), **kwargs)
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d_y_bottom, 1 + d_y_bottom), **kwargs)
            ax_bottom.plot((1 - d, 1 + d), (1 - d_y_bottom, 1 + d_y_bottom), **kwargs)
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.tick_params(axis='y', which='both', left=True)
        axes = [ax]
        main_ax = ax

    # Calculate section width (number of positions per graph)
    positions_per_graph = num_models
    
    # Keep track of graph section boundaries
    graph_centers = []
    separator_positions = []
    
    # Build boxplot_data & position_model_map
    boxplot_data = {}
    position_model_map = {}
    
    # Process each graph-model combination to gather data
    for g_idx, graph in enumerate(graphs):
        # Calculate position information - ensure equal spacing
        start_pos = g_idx * positions_per_graph
        graph_centers.append(start_pos + positions_per_graph/2 - 0.5)
        
        if g_idx > 0:
            separator_positions.append(start_pos - 0.5)
        
        # Process each model in the specified order
        for m_idx, model in enumerate(model_order):
            # Calculate position for this model
            position = start_pos + m_idx
            
            # Get relative gap values for this graph-model combination
            model_data = filtered_results[(filtered_results['graph']==graph) & 
                                        (filtered_results['model']==model)]
            
            # Store data even if empty (will handle in plotting)
            if not model_data.empty:
                boxplot_data[position] = model_data['rel_cut_gap'].tolist()
            else:
                boxplot_data[position] = []
                
            # Keep track of which model is at which position
            position_model_map[position] = model
    
    # Sort positions (to ensure they're in order)
    sorted_positions = sorted(boxplot_data.keys())
    
    # Prepare data for boxplot
    all_data = [boxplot_data[pos] for pos in sorted_positions]
    
    # Create a mask for positions with no data
    mask = [len(data) > 0 for data in all_data]
    valid_positions = [pos for i, pos in enumerate(sorted_positions) if mask[i]]
    valid_data = [data for data in all_data if len(data) > 0]
    
    for current_ax in axes:
        # Create boxplots for positions with data
        bp = current_ax.boxplot(
            valid_data,
            positions=valid_positions,
            patch_artist=True,
            widths=0.7,
            medianprops={'color': 'black'}
        )

        # Color each box according to its model
        for i, box in enumerate(bp['boxes']):
            model = position_model_map[valid_positions[i]]
            box.set_facecolor(get_model_color(model, model_order, color_order))

        # Enhance median lines for collapsed boxes
        y_min, y_max = current_ax.get_ylim()
        collapse_threshold = (y_max - y_min) * 0.01

        for i, box in enumerate(bp['boxes']):
            box_extent = box.get_path().get_extents()
            box_height = box_extent.height
            if box_height < collapse_threshold:
                model = position_model_map[valid_positions[i]]
                median_line = bp['medians'][i]
                median_line.set_linewidth(4)
                median_line.set_color(get_model_color(model, model_order, color_order))

        # Add vertical separators between graphs
        for pos in separator_positions:
            current_ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.7)
        
        # Draw horizontal line at 0% (optimal)
        current_ax.axhline(y=0, color='green', linestyle='-', linewidth=1, alpha=0.7)
    
    # Set x-axis labels and ticks
    main_ax = axes[-1] if use_broken_axis else ax
    main_ax.set_xticks(graph_centers)
    main_ax.set_xticklabels([f"{g}" for g in graphs])
    
    # Add y-axis label and title
    fig.text(0.02, 0.5, 'Relative optimality gap (%)', va='center', rotation='vertical')
    if ADD_TITLES:
        title_ax = axes[0]  # always use the top-most axis for the title
        title_ax.set_title(f'{SERIES_NAME} | Avg. rel. optimality gap distribution by graph (best hyperparams = {best_hyperparams_mode})')
    
    # Add legend for models
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=get_model_color(model, model_order, color_order),
                                    edgecolor='black') for model in model_order]

    # Add optimum line to legend
    opt_handle = Line2D([0], [0], color='green', linestyle='-', label='Optimum (0%)')
    legend_elements.append(opt_handle)
    
    legend_ax = axes[0]
    bbox_anchor = (1.02, 0.5)
    if use_broken_axis:
        # Adjust anchor to be in the middle of the combined axes, not just the top one.
        bbox_anchor = (1.02, 0.25)

    legend_ax.legend(
        legend_elements,
        model_order + ['Optimum (0%)'],
        loc='center left',
        bbox_to_anchor=bbox_anchor,
        borderaxespad=0
    )
    
    # Adjust layout and save
    plt.tight_layout(rect=[0.03, 0, 1, 1])
    if use_broken_axis:
        fig.subplots_adjust(hspace=0.05)

    save_figure('rel_gap_distributions', out_dir)
    plt.close()
    
    print(f"Saved relative optimality gap distribution plot to {os.path.join(out_dir, 'rel_gap_distributions.{FILE_EXT}')}")
    
def format_number(value):
    """Format a number to exactly 2 decimal places."""
    return f"{value:.2f}"

def plot_1d_hyperparam(data, param, model, ax, graph_name, is_leftmost=True, is_bottom=True, plot_idx=0, mode='mean_gap'):
    """Create a line plot of relative cut gap or success rate versus a single hyperparameter on a given axis."""
    # Map parameter names to LaTeX labels
    param_label_map = {
        'gamma_factor': r'$f_\gamma$',
        'r_target': r'$r_\mathrm{target}$',
        'poly_order': r'$n$',
        'alpha': r'$\alpha$'
    }
    
    # For QPDC, use special label for gamma_rate/gamma_th
    if model == 'q-PDC' and param == 'gamma_rate/gamma_th':
        param_label = r'Normalized rate of $\gamma$ increase, $\varepsilon_\gamma/\gamma_\mathrm{th}$'
    else:
        # Use the mapped label if available, otherwise use the parameter name
        param_label = param_label_map.get(param, param)
    
    # Group by the parameter and calculate metric based on mode
    if mode == 'success_rate':
        grouped = data.groupby(param)['reached_optimum_cut'].mean().reset_index()
        grouped['reached_optimum_cut'] = grouped['reached_optimum_cut'] * 100  # Convert to percentage
        metric_col = 'reached_optimum_cut'
        y_label = 'Success Rate (SR)'
        find_best = 'max'  # Higher is better for success rate
    elif mode == 'mean_gap':
        grouped = data.groupby(param)['rel_cut_gap'].mean().reset_index()
        metric_col = 'rel_cut_gap'
        y_label = 'Mean relative optimality gap (%)'
        find_best = 'min'  # Lower is better for gap
    else: # min_gap
        grouped = data.groupby(param)['rel_cut_gap'].min().reset_index()
        metric_col = 'rel_cut_gap'
        y_label = 'Minimum relative optimality gap (%)'
        find_best = 'min'  # Lower is better for gap

    # Sort by parameter value for proper line plot
    grouped = grouped.sort_values(param)
    
    # Create the line plot on the given axis
    ax.plot(grouped[param], grouped[metric_col], marker='o', linewidth=2)
    
    # Mark the best value (min for gap, max for success rate)
    if not grouped.empty:
        if find_best == 'min':
            best_idx = grouped[metric_col].idxmin()
        else:
            best_idx = grouped[metric_col].idxmax()
        best_point = grouped.loc[best_idx]
        ax.plot(best_point[param], best_point[metric_col], 'r*', markersize=12)
    
    if is_bottom:
        ax.set_xlabel(param_label)
    if is_leftmost:
        ax.set_ylabel(y_label)
    # Add subfigure identifier to the top-left
    ax.text(0.05, 1.1, graph_name, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='right')
    ax.grid(True)

def determine_decimal_places(values):
    """
    Determine the appropriate number of decimal places (0-2) based on the values.
    """
    # Convert to float to ensure consistent handling
    float_values = [float(v) for v in values]
    
    # Check if all values are effectively integers
    if all(abs(v - round(v)) < 1e-10 for v in float_values):
        return 0
    
    # Check if all values can be distinguished with 1 decimal place
    rounded_1dp = [round(v, 1) for v in float_values]
    if len(set(rounded_1dp)) == len(float_values):
        return 1
    
    # Default to 2 decimal places
    return 2

def format_tick_values(values):
    """Format values for plot ticks with appropriate number of decimal places (0-2)"""
    decimal_places = determine_decimal_places(values)
    
    if decimal_places == 0:
        return [f"{int(round(v))}" for v in values]
    elif decimal_places == 1:
        return [f"{v:.1f}" for v in values]
    else:
        return [f"{v:.2f}" for v in values]

def plot_2d_hyperparam(data, param1, param2, model, ax, graph_name, has_seaborn, global_min=None, global_max=None, is_leftmost=True, is_bottom=True, plot_idx=0, mode='mean_gap'):
    """Create a heatmap of relative cut gap or success rate versus two hyperparameters on a given axis."""
    # Map parameter names to LaTeX labels
    param_label_map = {
        'gamma_factor': r'$f_\gamma$',
        'r_target': r'$r_\mathrm{target}$',
        'poly_order': r'$n$',
        'alpha': r'$\alpha$',
        'B_num_vertices': r'$(B/A)n_\mathrm{vertices}$',
        'zeta': r'$\zeta$',
    }
    
    # Set parameter labels using the mapping
    if model == 'q-PDC' and param1 == 'gamma_rate/gamma_th':
        param1_label = r'Normalized rate of $\gamma$ increase ($\varepsilon_\gamma/\gamma_\mathrm{th}$)'
    else:
        param1_label = param_label_map.get(param1, param1)
        
    if model == 'q-PDC' and param2 == 'gamma_rate/gamma_th':
        param2_label = r'Normalized rate of $\gamma$ increase ($\varepsilon_\gamma/\gamma_\mathrm{th}$)'
    else:
        param2_label = param_label_map.get(param2, param2)
    
    # Group by both parameters and calculate metric based on mode
    if mode == 'success_rate':
        grouped = data.groupby([param1, param2])['reached_optimum_cut'].mean().reset_index()
        grouped['reached_optimum_cut'] = grouped['reached_optimum_cut'] * 100  # Convert to percentage
        metric_col = 'reached_optimum_cut'
        find_best = 'max'  # Higher is better for success rate
    elif mode == 'mean_gap':
        grouped = data.groupby([param1, param2])['rel_cut_gap'].mean().reset_index()
        metric_col = 'rel_cut_gap'
        find_best = 'min'  # Lower is better for gap
    else: # min_gap
        grouped = data.groupby([param1, param2])['rel_cut_gap'].min().reset_index()
        metric_col = 'rel_cut_gap'
        find_best = 'min'  # Lower is better for gap

    # Get unique values for each parameter, sorted
    param1_values = sorted(grouped[param1].unique())
    param2_values = sorted(grouped[param2].unique())
    
    # Format tick labels with appropriate decimal places
    param1_tick_labels = format_tick_values(param1_values)
    param2_tick_labels = format_tick_values(param2_values)
    
    # Determine which parameter has fewer unique values - use that as columns
    if len(param1_values) <= len(param2_values):
        # param1 has fewer values, use it as columns
        x_param, y_param = param1, param2
        x_label, y_label = param1_label, param2_label
        x_values, y_values = param1_values, param2_values
        x_tick_labels, y_tick_labels = param1_tick_labels, param2_tick_labels
    else:
        # param2 has fewer values, use it as columns
        x_param, y_param = param2, param1
        x_label, y_label = param2_label, param1_label
        x_values, y_values = param2_values, param1_values
        x_tick_labels, y_tick_labels = param2_tick_labels, param1_tick_labels
    
    # Create a pivot table for the heatmap with fewer points as columns
    pivot = grouped.pivot_table(index=y_param, columns=x_param, values=metric_col)
    
    if has_seaborn:
        # Use seaborn for a nicer heatmap
        import seaborn as sns
        
        # Use a logarithmic color bar for gap metrics, linear for success rate
        if mode == 'success_rate':
            norm = None
        else:
            norm = None
            if pd.notna(global_min) and global_min < global_max:
                norm = LogNorm(vmin=global_min, vmax=global_max)

        sns.heatmap(pivot, cmap='viridis', annot=False, fmt=".2f",
                    norm=norm,
                    cbar=False, ax=ax)
        
        # Clear default labels set by seaborn
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Mark the best value (min for gap, max for success rate)
        if find_best == 'min':
            best_val = pivot.min().min()
        else:
            best_val = pivot.max().max()
        if pd.notna(best_val):
            if find_best == 'min':
                best_pos = pivot.stack().idxmin()
            else:
                best_pos = pivot.stack().idxmax()
            y_best_idx = pivot.index.get_loc(best_pos[0])
            x_best_idx = pivot.columns.get_loc(best_pos[1])
            ax.add_patch(plt.Rectangle((x_best_idx, y_best_idx), 1, 1, fill=False, edgecolor='red', lw=2))

        # Adjust only the top/bottom limits without adding extra padding
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top)
        
        # Set formatted tick labels
        if len(x_tick_labels) > 10:
            step = max(1, int(np.ceil(len(x_tick_labels) / 10)))
            x_tick_indices = np.arange(0, len(x_tick_labels), step)
            x_ticks_subset = x_tick_indices + 0.5
            x_labels_subset = [x_tick_labels[i] for i in x_tick_indices]
            ax.set_xticks(x_ticks_subset)
            ax.set_xticklabels(x_labels_subset, rotation=0)
        else:
            ax.set_xticks(np.arange(len(x_tick_labels)) + 0.5)
            ax.set_xticklabels(x_tick_labels, rotation=0)
        
        # Reduce number of y-ticks if there are too many
        if len(y_tick_labels) > 10:
            step = max(1, int(np.ceil(len(y_tick_labels) / 10)))
            y_tick_indices = np.arange(0, len(y_tick_labels), step)
            y_ticks_subset = y_tick_indices + 0.5
            y_labels_subset = [y_tick_labels[i] for i in y_tick_indices]
            ax.set_yticks(y_ticks_subset)
            ax.set_yticklabels(y_labels_subset, rotation=0)
        else:
            ax.set_yticks(np.arange(len(y_tick_labels)) + 0.5)
            ax.set_yticklabels(y_tick_labels, rotation=0)
    else:
        # Use matplotlib's imshow for the heatmap
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower')
        cbar_label = 'Success Rate (SR)' if mode == 'success_rate' else 'Mean relative optimality gap (%)'
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
        
        # Set formatted tick labels
        if len(x_values) > 10:
            step = max(1, int(np.ceil(len(x_values) / 10)))
            x_tick_indices = np.arange(0, len(x_values), step)
            x_labels_subset = [x_tick_labels[i] for i in x_tick_indices]
            ax.set_xticks(x_tick_indices, x_labels_subset, rotation=0)
        else:
            ax.set_xticks(range(len(x_values)), x_tick_labels, rotation=0)
        
        # Reduce number of y-ticks if there are too many
        if len(y_values) > 10:
            step = max(1, int(np.ceil(len(y_values) / 10)))
            y_tick_indices = np.arange(0, len(y_values), step)
            y_labels_subset = [y_tick_labels[i] for i in y_tick_indices]
            ax.set_yticks(y_tick_indices, y_labels_subset)
        else:
            ax.set_yticks(range(len(y_values)), y_tick_labels)
    
    if is_bottom:
        ax.set_xlabel(x_label)
    if is_leftmost:
        ax.set_ylabel(y_label)
    # Add subfigure identifier to the top-left
    ax.text(0.05, 1.1, graph_name, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='right')

if __name__ == "__main__":
    main()
