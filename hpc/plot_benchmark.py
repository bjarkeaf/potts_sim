#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from collections import defaultdict

# CLI flags:
#   --results     Path to the results.parquet file (required)
#   --out_dir     Directory where plots will be saved (default: plots)
#   --model_order Comma-separated list of model names in the desired legend/order

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from Potts model parameter sweep results')
    parser.add_argument('--results', type=str, required=True, help='Path to results.parquet file')
    parser.add_argument('--out_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--model_order', type=str, default=None, help='Comma-separated list of models in desired order')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load results
    results = pd.read_parquet(args.results)
    
    # Alias mapping for certain model keys to nicer labels
    alias_map = {
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
    
    # Calculate optimality gap if not already present
    if 'energy_gap' not in results.columns and 'energy' in results.columns and 'opt_energy' in results.columns:
        results['energy_gap'] = results['energy'] - results['opt_energy']
    
    raw_order = args.model_order.split(',') if args.model_order else None
    model_order = [alias_map.get(m, m) for m in raw_order] if raw_order else None

    # 1. Max success rate for each model versus graph
    if 'reached_optimum_cut' in results.columns:
        plot_max_success_rate_by_graph(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create success rate plot - required columns not found")
    
    # 2. Absolute & relative optimality gap
    if 'energy_gap' in results.columns:
        plot_energy_gap_by_graph(results, args.out_dir, model_order)
        # simple relative gap (%)
        results['rel_energy_gap'] = results['energy_gap'] / results['opt_energy'] * 100
        plot_relative_energy_gap_by_graph(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create optimality gap plots - required columns not found")
    
    # 3. Number of problems with max success rate per model (sorted bar plot)
    if 'reached_optimum_cut' in results.columns:
        plot_models_by_max_success_count(results, args.out_dir, model_order)
    else:
        print("Warning: Cannot create max success count plot - required columns not found")

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

def plot_energy_gap_by_graph(results, out_dir, model_order=None):
    """Plot average optimality gap from ground state for each model versus number of spins"""
    # Group by graph and model to calculate average optimality gap
    avg_energy_gap = results.groupby(['graph', 'model'])['energy_gap'].mean().reset_index()
    
    # Add num_spins information by merging with results
    graph_spins = results[['graph', 'num_spins']].drop_duplicates()
    avg_energy_gap = avg_energy_gap.merge(graph_spins, on='graph')
    
    # Get all models
    models = model_order if model_order else sorted(avg_energy_gap['model'].unique())
    
    # Get unique spin counts
    spin_counts = sorted(graph_spins['num_spins'].unique())
    
    # Plot average optimality gap by number of spins for each model
    plt.figure(figsize=(7, 3))
    
    # Define bar positions
    bar_positions = np.arange(len(spin_counts))
    
    # Calculate optimality gaps by spin count and model
    diff_by_spin_model = {}
    model_order_by_spin = {}
    
    for spin_count in spin_counts:
        # Calculate optimality gap for each model for this spin count
        diff_by_model = {}
        for model in models:
            model_data = avg_energy_gap[(avg_energy_gap['model'] == model) & 
                                         (avg_energy_gap['num_spins'] == spin_count)]
            diff = model_data['energy_gap'].mean() if not model_data.empty else 0
            diff_by_model[model] = diff
        
        # Store differences for this spin count
        diff_by_spin_model[spin_count] = diff_by_model
        
        # Sort models by optimality gap (ascending order - shorter to taller)
        model_order_by_spin[spin_count] = sorted(models, key=lambda m: diff_by_model[m])
    
    # Plot bars for each spin count with models ordered by height
    width = 0.8 / len(models)
    
    for i, spin_count in enumerate(spin_counts):
        ordered_models = model_order_by_spin[spin_count]
        
        for j, model in enumerate(ordered_models):
            diff = diff_by_spin_model[spin_count][model]
            offset = (j - (len(models) - 1) / 2) * width
            
            # Use a consistent color for each model across all spin counts
            model_idx = models.index(model)
            plt.bar(bar_positions[i] + offset, diff, width, 
                   color=plt.cm.tab10(model_idx % 10), 
                   label=model if i == 0 and j == 0 else "")
    
    # Create a proper legend with all models
    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(models.index(model) % 10)) for model in models]
    plt.legend(handles, models)
    
    plt.xlabel('Number of Spins')
    plt.ylabel('Average optimality gap\n'+r'($H - H_\mathrm{GS}$)')
    #plt.title('Average optimality gap from Ground State by Model and Problem Size')
    plt.xticks(bar_positions, spin_counts)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'avg_energy_gap.png'), dpi=200)
    print(f"Saved average optimality gap plot to {os.path.join(out_dir, 'avg_energy_gap.png')}")

def plot_relative_energy_gap_by_graph(results, out_dir, model_order=None):
    """Plot average relative optimality gap (%) for each model versus number of spins"""
    # compute mean relative gap
    avg_rel = results.groupby(['graph','model'])['rel_energy_gap'].mean().reset_index()
    # merge spin counts
    graph_spins = results[['graph','num_spins']].drop_duplicates()
    avg_rel = avg_rel.merge(graph_spins, on='graph')
    # determine model order and spins
    models = model_order if model_order else sorted(avg_rel['model'].unique())
    spin_counts = sorted(graph_spins['num_spins'].unique())
    # setup plot
    plt.figure(figsize=(6,3))
    bar_positions = np.arange(len(spin_counts))
    # calculate and sort bars
    diff_by_spin = {}
    order_by_spin = {}
    for s in spin_counts:
        dm = {m: avg_rel[(avg_rel.model==m)&(avg_rel.num_spins==s)]['rel_energy_gap'].mean() 
              if not avg_rel[(avg_rel.model==m)&(avg_rel.num_spins==s)].empty else 0
              for m in models}
        diff_by_spin[s] = dm
        order_by_spin[s] = sorted(models, key=lambda m: dm[m])
    width = 0.8/len(models)
    for i,s in enumerate(spin_counts):
        for j,m in enumerate(order_by_spin[s]):
            off = (j-(len(models)-1)/2)*width
            plt.bar(bar_positions[i]+off, diff_by_spin[s][m], width,
                    color=plt.cm.tab10(models.index(m)%10),
                    label=m if i==0 and j==0 else "")
    # legend, labels, save
    handles = [plt.Rectangle((0,0),1,1,color=plt.cm.tab10(models.index(m)%10)) for m in models]
    plt.legend(
        handles, models,
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        borderaxespad=0.0
    )
    plt.xlabel('Number of Spins')
    plt.ylabel('Average optimality gap (%)')
    plt.xticks(bar_positions, spin_counts)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'rel_avg_energy_gap.png'), dpi=200)
    print(f"Saved relative optimality gap plot to {os.path.join(out_dir,'rel_avg_energy_gap.png')}")

def plot_models_by_max_success_count(results, out_dir, model_order=None):
    """Plot number of problems with max success rate per model"""
    # Find max success rate for each graph-model combination
    success_rates = results.groupby(['graph', 'model', 'param_id'])['reached_optimum_cut'].mean().reset_index()
    max_success_by_model = success_rates.groupby(['graph', 'model'])['reached_optimum_cut'].max().reset_index()
    
    # Find which model has the max success rate for each graph
    max_success_by_graph = max_success_by_model.groupby('graph')['reached_optimum_cut'].max().reset_index()
    max_success_by_graph.rename(columns={'reached_optimum_cut': 'max_success_rate'}, inplace=True)
    
    # Identify models that achieve max success rate for each graph
    max_success_by_model = max_success_by_model.merge(max_success_by_graph, on='graph')
    max_models = max_success_by_model[max_success_by_model['reached_optimum_cut'] == max_success_by_model['max_success_rate']]
    
    # Count how many times each model achieves max success rate
    model_counts = max_models.groupby('model').size().reset_index(name='count')
    if model_order:
        import pandas as _pd  # ensure pd is available
        model_counts['model'] = _pd.Categorical(model_counts['model'],
                                               categories=model_order,
                                               ordered=True)
        model_counts = model_counts.sort_values('model')
    
    # Sort by count in descending order
    model_counts = model_counts.sort_values('count', ascending=False)
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create bars with different colors
    bars = plt.bar(model_counts['model'], model_counts['count'])
    
    # Add count on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, 
                 str(int(height)), ha='center', va='bottom')
    
    plt.ylabel('Number of problems\nwith highest SR')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'models_by_max_success_count.png'), dpi=200)
    print(f"Saved max success count plot to {os.path.join(out_dir, 'models_by_max_success_count.png')}")

if __name__ == "__main__":
    main()
