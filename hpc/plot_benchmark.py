#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from collections import defaultdict

SERIES_NAME = None

# CLI flags:
#   --results     Path to the results.parquet file (required)
#   --out_dir     Directory where plots will be saved (default: plots/<series_name>)
#   --model_order Comma-separated list of model names in the desired legend/order

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from Potts model parameter sweep results')
    parser.add_argument('--results', type=str, required=True, help='Path to results.parquet file')
    parser.add_argument('--out_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--model_order', type=str, default=None, help='Comma-separated list of models in desired order')
    args = parser.parse_args()
    
    # infer series name and adjust default output directory
    input_path = Path(args.results)
    if input_path.name.startswith('results_') and input_path.suffix == '.parquet':
        series_name = input_path.stem[len('results_'):]
    else:
        series_name = input_path.stem
    global SERIES_NAME
    SERIES_NAME = series_name
    default_out = parser.get_default('out_dir')
    if args.out_dir == default_out:
        args.out_dir = str(Path(default_out) / series_name)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load results
    results = pd.read_parquet(args.results)
    
    # Alias mapping for certain model keys to nicer labels
    alias_map = {
        'QPDC':         'q-PDC',
        'POLYNOMIAL':      'Polynomial',
        'NEC':             'NEC',
        'SIGMOID':         'Sigmoid',
        'FIXED_AMPLITUDE': 'q-SHIL'
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
    if 'reached_optimum_cut' in results.columns:
        plot_max_success_rate_by_graph(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create success rate plot - required columns not found")
        
    # 3. Absolute & relative cut gap
    if 'cut_gap' in results.columns:
        plot_optimality_gap(results, args.out_dir, model_order)
        plot_relative_optimality_gap(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create optimality gap plots - cut value columns not found")
    
    # 4. Hyperparameter sweep plots for cut gap (not energy gap)
    if 'cut_gap' in results.columns:
        plot_hyperparams(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create hyperparameter sweep plots - cut gap not found")

def sort_models_by_performance(data, metric_col, ascending=True):
    """
    Sort models by their average performance on the given metric.
    
    Parameters:
    - data: DataFrame with results
    - metric_col: Column name of the metric to use for sorting
    - ascending: If True, sort in ascending order (lower is better)
    
    Returns:
    - List of model names sorted by performance
    """
    # Calculate average metric value for each model
    avg_by_model = data.groupby('model')[metric_col].mean().reset_index()
    
    # Sort models by average metric value
    sorted_models = avg_by_model.sort_values(metric_col, ascending=ascending)['model'].tolist()
    
    return sorted_models

def get_best_params_by_graph_model(results, metric_col, is_minimize=True):
    """
    Get the best parameter combination for each graph-model pair based on the specified metric.
    Returns a DataFrame with best parameter IDs for each graph-model combination.
    
    Parameters:
    - results: DataFrame with results
    - metric_col: Column name of the metric to optimize
    - is_minimize: If True, minimize the metric; if False, maximize the metric
    """
    # Group by graph, model, and param_id to calculate mean metric
    grouped = results.groupby(['graph', 'model', 'param_id'])[metric_col].mean().reset_index()
    
    # Find the best param_id for each graph-model combination
    if is_minimize:
        best_params = grouped.loc[grouped.groupby(['graph', 'model'])[metric_col].idxmin()]
    else:
        best_params = grouped.loc[grouped.groupby(['graph', 'model'])[metric_col].idxmax()]
    
    # Return only the necessary columns
    return best_params[['graph', 'model', 'param_id']]

def plot_optimality_gap(results, out_dir, model_order=None):
    """Plot best average optimality gap per model as horizontal bars."""
    # Get best parameters for each graph-model combination (minimize cut gap)
    best_params = get_best_params_by_graph_model(results, 'cut_gap', is_minimize=True)
    
    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results, 
        best_params, 
        on=['graph', 'model', 'param_id']
    )
    
    # compute mean cut per model
    mean_cut = filtered_results.groupby('model')['cut_gap'].mean().sort_values()
    models = mean_cut.index.tolist()
    values = mean_cut.values

    plt.figure(figsize=(4, len(models)*0.6))
    plt.title(f'{SERIES_NAME} | Absolute optimality gap')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    max_val = values.max()
    offset = max_val * 0.01
    plt.barh(models, values, color=plt.cm.tab10(range(len(models))))
    for i, v in enumerate(values):
        plt.text(v + offset, i, f'{v:.2f}', va='center', ha='left')
    plt.xlabel('Average optimality gap (cut value)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'optimality_gaps.png'), dpi=200)
    plt.close()
    print(f"Saved best average optimality gap plot to {os.path.join(out_dir, 'optimality_gaps.png')}")

def plot_relative_optimality_gap(results, out_dir, model_order=None):
    """Plot best average relative optimality gap per model as horizontal bars."""
    # Get best parameters for each graph-model combination (minimize relative cut gap)
    best_params = get_best_params_by_graph_model(results, 'rel_cut_gap', is_minimize=True)
    
    # Filter results to only include best parameter combinations
    filtered_results = pd.merge(
        results, 
        best_params, 
        on=['graph', 'model', 'param_id']
    )
    
    # compute mean relative gap
    mean_rel_cut = filtered_results.groupby('model')['rel_cut_gap'].mean().sort_values()
    models = mean_rel_cut.index.tolist()
    values = mean_rel_cut.values

    plt.figure(figsize=(4, len(models)*0.6))
    plt.title(f'{SERIES_NAME} | Relative optimality gap')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    max_val = values.max()
    offset = max_val * 0.01
    plt.barh(models, values, color=plt.cm.tab10(range(len(models))))
    for i, v in enumerate(values):
        plt.text(v + offset, i, f'{v:.2f}%', va='center', ha='left')
    plt.xlabel('Relative optimality gap (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rel_optimality_gaps.png'), dpi=200)
    plt.close()
    print(f"Saved best relative optimality gap plot to {os.path.join(out_dir, 'rel_optimality_gaps.png')}")

def plot_hyperparams(results, out_dir, model_order=None):
    """
    Plot the average relative optimality gap versus hyperparameters for each model.
    
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
            
        # Identify which hyperparameters are being swept for this model
        swept_params = identify_swept_hyperparameters(model_data)
        
        # Based on number of swept parameters, create appropriate plot
        if len(swept_params) == 0:
            print(f"Skipping hyperparameter plot for {model}: No parameters being swept")
        elif len(swept_params) == 1:
            param = swept_params[0]
            print(f"Creating 1D hyperparameter plot for {model}: {param}")
            plot_1d_hyperparam(model_data, param, model, out_dir)
        elif len(swept_params) == 2:
            param1, param2 = swept_params
            print(f"Creating 2D hyperparameter plot for {model}: {param1} vs {param2}")
            plot_2d_hyperparam(model_data, param1, param2, model, out_dir, has_seaborn)
        else:
            print(f"Skipping hyperparameter plot for {model}: More than 2 parameters being swept ({swept_params})")

def identify_swept_hyperparameters(model_data):
    """
    Identify which hyperparameters are being swept.
    Returns a list of parameter names that have multiple unique values.
    """
    # List of potential hyperparameters to check
    potential_params = [
        'poly_order', 'gamma_factor', 'beta_factor', 
        'alpha_rate', 'r_target', 'alpha',
        'B_num_vertices', 'zeta'
    ]
    
    # Check which parameters have multiple unique values
    swept_params = []
    for param in potential_params:
        if param in model_data.columns and len(model_data[param].unique()) > 1:
            swept_params.append(param)
    
    return swept_params

def plot_1d_hyperparam(data, param, model, out_dir):
    """Create a line plot of relative cut gap versus a single hyperparameter."""
    plt.figure(figsize=(8, 5))
    
    # Group by the parameter and calculate mean relative cut gap
    grouped = data.groupby(param)['rel_cut_gap'].mean().reset_index()
    
    # Sort by parameter value for proper line plot
    grouped = grouped.sort_values(param)
    
    # Create the line plot
    plt.plot(grouped[param], grouped['rel_cut_gap'], marker='o', linewidth=2)
    
    plt.xlabel(param)
    plt.ylabel('Average optimality gap (%)')
    plt.title(f'{SERIES_NAME} | {model}: Cut gap vs. {param}')
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    filename = f'{model.lower().replace(" ", "_").replace("-", "_")}_1d_hyperparam.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()
    
    print(f"Saved 1D hyperparameter cut gap plot to {os.path.join(out_dir, filename)}")

def plot_2d_hyperparam(data, param1, param2, model, out_dir, has_seaborn):
    """Create a heatmap of relative cut gap versus two hyperparameters."""
    # Group by both parameters and calculate mean relative cut gap
    grouped = data.groupby([param1, param2])['rel_cut_gap'].mean().reset_index()
    
    # Get unique values for each parameter, sorted
    param1_values = sorted(grouped[param1].unique())
    param2_values = sorted(grouped[param2].unique())
    
    # Determine which parameter has fewer unique values - use that as columns
    if len(param1_values) <= len(param2_values):
        # param1 has fewer values, use it as columns
        x_param, y_param = param1, param2
        x_values, y_values = param1_values, param2_values
    else:
        # param2 has fewer values, use it as columns
        x_param, y_param = param2, param1
        x_values, y_values = param2_values, param1_values
    
    # Create a pivot table for the heatmap with fewer points as columns
    pivot = grouped.pivot_table(index=y_param, columns=x_param, values='rel_cut_gap')
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    if has_seaborn:
        # Use seaborn for a nicer heatmap
        import seaborn as sns
        ax = sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".2f",
                         cbar_kws={'label': 'Average optimality gap (%)'})
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
    else:
        # Use matplotlib's imshow for the heatmap
        im = plt.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(im, label='Average optimality gap (%)')
        plt.xticks(range(len(x_values)), x_values)
        plt.yticks(range(len(y_values)), y_values)
    
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'{SERIES_NAME} | {model}: Cut gap vs. {y_param} and {x_param}')
    
    # Save the plot
    plt.tight_layout()
    filename = f'{model.lower().replace(" ", "_").replace("-", "_")}_2d_hyperparam.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()
    
    print(f"Saved 2D hyperparameter optimality gap plot to {os.path.join(out_dir, filename)}")

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
                   color=plt.cm.tab10(model_idx % 10), 
                   label=model if i == 0 and j == 0 else "")
    
    # Create a proper legend with all models
    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(models.index(model) % 10)) for model in models]
    plt.legend(handles, models)
    
    plt.xlabel('Number of Spins')
    plt.ylabel('Success Rate (%)')
    #plt.title('Maximum Success Rate by Model and Problem Size')
    plt.xticks(bar_positions, spin_counts)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'max_success_rate.png'), dpi=200)
    print(f"Saved max success rate plot to {os.path.join(out_dir, 'max_success_rate.png')}")

if __name__ == "__main__":
    main()
