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
from matplotlib.ticker import FormatStrFormatter
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

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from Potts model parameter sweep results')
    parser.add_argument('--results', type=str, required=True, help='Path to results.parquet file')
    parser.add_argument('--out_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--model_order', type=str, default=None, help='Comma-separated list of models in desired order')
    parser.add_argument('--figure_mode', action='store_true', help='Output as PDF without titles (for publications)')
    parser.add_argument('--table', action='store_true', help='Generate CSV table with statistics for best hyperparameter combinations')
    parser.add_argument('--best_hyperparams', type=str, default='mean_gap', choices=['mean_gap', 'min_gap'], help='Metric for selecting best hyperparameters: mean_gap (minimize mean) or min_gap (minimize minimum).')
    args = parser.parse_args()
    
    # Set global variables for figure mode
    global FILE_EXT, ADD_TITLES
    if args.figure_mode:
        FILE_EXT = "pdf"
        ADD_TITLES = True
    
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
        'CIM':              'Sigmoid IM'
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

    # 1. Max success rate for each model versus graph
    # if 'reached_optimum_cut' in results.columns:
    #     plot_max_success_rate_by_graph(results, args.out_dir, model_order)
    # else:
    #     print("Warning: Cannot create success rate plot - required columns not found")
        
    # 3. Absolute & relative cut gap
    # if 'cut_gap' in results.columns:
    #     plot_optimality_gap(results, args.out_dir, model_order)
    #     plot_relative_optimality_gap(results, args.out_dir, model_order, args.best_hyperparams)
    # else:
    #     print("Warning: Cannot create optimality gap plots - cut value columns not found")
    
    # 4. Hyperparameter sweep plots for cut gap (not energy gap)
    if 'cut_gap' in results.columns:
        plot_hyperparams(results, args.out_dir, model_order, args.best_hyperparams)
    else:
        print("Warning: Cannot create hyperparameter sweep plots - cut gap not found")
    
    # 6. Relative optimality gap distribution boxplots by graph and model
    if 'rel_cut_gap' in results.columns:
        plot_rel_gap_distributions_by_graph(results, args.out_dir, model_order, args.best_hyperparams)
    else:
        print("Warning: Cannot create relative gap distribution plots - relative cut gap not found")

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
    For QPDC model, add gamma_rate/gamma_th column which is gamma_factor/T
    Returns a modified copy of the dataframe
    """
    if 'gamma_factor' in data.columns and 'T' in data.columns:
        data = data.copy()
        data['gamma_rate/gamma_th'] = data['gamma_factor'] / data['T']
    return data

def plot_hyperparams(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap'):
    """
    Plot the mean relative optimality gap versus hyperparameters for each model.
    
    For 1D hyperparameter sweeps: Line plot of gap vs parameter
    For 2D hyperparameter sweeps: Heatmap with parameters as axes
    For 3D+ hyperparameter sweeps: Skip plotting with a message
    For no hyperparameter sweep: Skip plotting with a message
    """
    # Try to import seaborn for better heatmaps
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
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
            if best_hyperparams_mode == 'mean_gap':
                all_graph_data = model_data.groupby([param1, param2])['rel_cut_gap'].mean()
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
            log_norm = None
            if pd.notna(global_min) and global_min < global_max:
                log_norm = LogNorm(vmin=global_min, vmax=global_max)
            
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=log_norm)
            sm.set_array([])
            
            # Position colorbar at the bottom
            cbar_ax = fig.add_axes([0.15, 0.065, 0.7, 0.02]) # change second for bottom margin
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Mean relative optimality gap (%)')

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

def plot_max_success_rate_by_graph(results, out_dir, model_order=None):
    """Plot max success rate for each model versus graph, ordered by number of spins and success rate"""
    # Group by graph, model, and param_id to calculate success rate
    success_rates = results.groupby(['graph', 'model', 'param_id'])['reached_optimum_cut'].mean().reset_index()
    
    # Find max success rate for each graph-model combination
    max_success_rates = success_rates.groupby(['graph', 'model'])['reached_optimum_cut'].max().reset_index()
    max_success_rates.rename(columns={'reached_optimum_cut': 'success_rate'}, inplace=True)
    
    # Add num_spins information by merging with results
    graph_spins = results[['graph', 'num_spins']].drop_duplicates()
    max_success_rates = max_success_rates.merge(graph_spins, on='graph')
    
    # Get all models
    models = model_order if model_order else sorted(max_success_rates['model'].unique())
    
    # Get unique spin counts
    spin_counts = sorted(graph_spins['num_spins'].unique())
    
    # Plot max success rate by number of spins for each model
    plt.figure(figsize=(7, 3))
    if ADD_TITLES:
        plt.title(f'{SERIES_NAME} | Max success rate')
    
    # Define bar positions
    bar_positions = np.arange(len(spin_counts))
    
    # Calculate success rates by spin count and model
    sr_by_spin_model = {}
    model_order_by_spin = {}
    
    for spin_count in spin_counts:
        # Calculate success rate for each model for this spin count
        sr_by_model = {}
        for model in models:
            model_data = max_success_rates[(max_success_rates['model'] == model) & 
                                          (max_success_rates['num_spins'] == spin_count)]
            sr = model_data['success_rate'].mean() * 100 if not model_data.empty else 0
            sr_by_model[model] = sr
        
        # Store success rates for this spin count
        sr_by_spin_model[spin_count] = sr_by_model
        
        # Sort models by success rate (ascending order - shorter to taller)
        model_order_by_spin[spin_count] = sorted(models, key=lambda m: sr_by_model[m])
    
    # Plot bars for each spin count with models ordered by height
    width = 0.8 / len(models)
    
    for i, spin_count in enumerate(spin_counts):
        ordered_models = model_order_by_spin[spin_count]
        
        for j, model in enumerate(ordered_models):
            sr = sr_by_spin_model[spin_count][model]
            offset = (j - (len(models) - 1) / 2) * width
            
            # Use a consistent color for each model across all spin counts
            model_idx = models.index(model)
            plt.bar(bar_positions[i] + offset, sr, width,
                   color=PALETTE[model_idx % len(PALETTE)],
                   label=model if i == 0 and j == 0 else "")
    
    # Create a proper legend with all models
    handles = [plt.Rectangle((0,0),1,1, color=PALETTE[i % len(PALETTE)])
               for i, _ in enumerate(models)]
    plt.legend(handles, models)
    
    plt.xlabel('Number of Spins')
    plt.ylabel('Success Rate (%)')
    plt.xticks(bar_positions, spin_counts)
    plt.tight_layout()
    save_figure('max_success_rate', out_dir)
    plt.close()

def plot_rel_gap_distributions_by_graph(results, out_dir, model_order=None, best_hyperparams_mode='mean_gap'):
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
    graphs_df = filtered_results[['graph', 'num_spins']].drop_duplicates()
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
        for i in range(len(axes)-1):
            ax_top = axes[i]
            ax_bottom = axes[i+1]
            ratio = height_ratios[i+1] / height_ratios[i]
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
            ax_top.plot((-d, +d), (-d*ratio, +d*ratio), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d*ratio, +d*ratio), **kwargs)
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
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
            model_idx = model_order.index(model)
            box.set_facecolor(PALETTE[model_idx % len(PALETTE)])
        
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
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=PALETTE[i % len(PALETTE)], 
                                    edgecolor='black') for i, _ in enumerate(model_order)]
    
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
    """Create a line plot of relative cut gap versus a single hyperparameter on a given axis."""
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
    
    # Group by the parameter and calculate mean or min relative cut gap
    if mode == 'mean_gap':
        grouped = data.groupby(param)['rel_cut_gap'].mean().reset_index()
        metric_col = 'rel_cut_gap'
        y_label = 'Mean relative optimality gap (%)'
    else: # min_gap
        grouped = data.groupby(param)['rel_cut_gap'].min().reset_index()
        metric_col = 'rel_cut_gap'
        y_label = 'Minimum relative optimality gap (%)'

    # Sort by parameter value for proper line plot
    grouped = grouped.sort_values(param)
    
    # Create the line plot on the given axis
    ax.plot(grouped[param], grouped[metric_col], marker='o', linewidth=2)
    
    # Mark the minimum value
    if not grouped.empty:
        min_idx = grouped[metric_col].idxmin()
        min_point = grouped.loc[min_idx]
        ax.plot(min_point[param], min_point[metric_col], 'r*', markersize=12)
    
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
    """Create a heatmap of relative cut gap versus two hyperparameters on a given axis."""
    # Map parameter names to LaTeX labels
    param_label_map = {
        'gamma_factor': r'$f_\gamma$',
        'r_target': r'$r_\mathrm{target}$',
        'poly_order': r'$n$',
        'alpha': r'$\alpha$',
        'B_num_vertices': r'$(B/A)n_\mathrm{vertices}$',
        'zeta': r'$\zeta$',
    }
    # param_label_map = {
    #     'gamma_factor': r'Phase discretization vs. coupling factor, $f_\gamma$',
    #     'r_target': r'Target amplitude, $r_\mathrm{target}$',
    #     'poly_order': r'Nonlinearity order, $n$',
    #     'alpha': r'Amplitude gain, $\alpha$',
    #     'B_num_vertices': r'Edge- vs. one-hot constraint factor, $(B/A)n_\mathrm{vertices}$',
    #     'zeta': r'External field scaling factor, $\zeta$',
    # }
    
    # Set parameter labels using the mapping
    if model == 'q-PDC' and param1 == 'gamma_rate/gamma_th':
        param1_label = r'Normalized rate of $\gamma$ increase ($\varepsilon_\gamma/\gamma_\mathrm{th}$)'
    else:
        param1_label = param_label_map.get(param1, param1)
        
    if model == 'q-PDC' and param2 == 'gamma_rate/gamma_th':
        param2_label = r'Normalized rate of $\gamma$ increase ($\varepsilon_\gamma/\gamma_\mathrm{th}$)'
    else:
        param2_label = param_label_map.get(param2, param2)
    
    # Group by both parameters and calculate mean or min relative cut gap
    if mode == 'mean_gap':
        grouped = data.groupby([param1, param2])['rel_cut_gap'].mean().reset_index()
    else: # min_gap
        grouped = data.groupby([param1, param2])['rel_cut_gap'].min().reset_index()

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
    pivot = grouped.pivot_table(index=y_param, columns=x_param, values='rel_cut_gap')
    
    if has_seaborn:
        # Use seaborn for a nicer heatmap
        import seaborn as sns
        
        # Use a logarithmic color bar to better resolve low-value regions
        log_norm = None
        if pd.notna(global_min) and global_min < global_max:
            log_norm = LogNorm(vmin=global_min, vmax=global_max)

        sns.heatmap(pivot, cmap='viridis', annot=False, fmt=".2f",
                    norm=log_norm,
                    cbar=False, ax=ax)
        
        # Clear default labels set by seaborn
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Mark the minimum value
        min_val = pivot.min().min()
        if pd.notna(min_val):
            min_pos = pivot.stack().idxmin()
            y_min_idx = pivot.index.get_loc(min_pos[0])
            x_min_idx = pivot.columns.get_loc(min_pos[1])
            ax.add_patch(plt.Rectangle((x_min_idx, y_min_idx), 1, 1, fill=False, edgecolor='red', lw=2))

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
            ax.set_xticklabels(x_tick_labels, rotation=0)  # Set rotation to 0 to avoid extra spacing
        
        # Reduce number of y-ticks if there are too many
        if len(y_tick_labels) > 10:
            step = max(1, int(np.ceil(len(y_tick_labels) / 10)))
            y_tick_indices = np.arange(0, len(y_tick_labels), step)
            y_ticks_subset = y_tick_indices + 0.5
            y_labels_subset = [y_tick_labels[i] for i in y_tick_indices]
            ax.set_yticks(y_ticks_subset)
            ax.set_yticklabels(y_labels_subset, rotation=0)
        else:
            # Explicitly set y-ticks to match labels to avoid seaborn/matplotlib bug
            ax.set_yticks(np.arange(len(y_tick_labels)) + 0.5)
            ax.set_yticklabels(y_tick_labels, rotation=0)
    else:
        # Use matplotlib's imshow for the heatmap
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower')
        ax.figure.colorbar(im, ax=ax, label='Mean relative optimality gap (%)')
        
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