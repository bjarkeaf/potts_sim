#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='Analyze Potts model parameter sweep results')
    parser.add_argument('--results', type=str, required=True, help='Path to results.parquet file')
    parser.add_argument('--out-dir', type=str, default='analysis', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load results
    results = pd.read_parquet(args.results)
    
    print(f"Loaded {len(results)} results")
    print(f"Models: {results['model'].unique()}")
    print(f"Graphs: {results['graph'].unique()}")
    
    # Basic statistics
    print("\nOverall Statistics:")
    print(f"Mean cut_diff from optimum: {results['cut_diff'].mean():.2f}")
    print(f"Best cut_diff from optimum: {results['cut_diff'].min():.2f}")
    
    # Group by model and calculate statistics
    model_stats = results.groupby('model').agg({
        'cut_diff': ['mean', 'min', 'std'],
        'runtime': ['mean', 'std']
    })
    
    print("\nModel Statistics:")
    print(model_stats)
    
    # Group by model and param_id to find best parameters for each model
    param_stats = results.groupby(['model', 'param_id']).agg({
        'cut_diff': ['mean', 'min', 'std'],
        'runtime': 'mean'
    }).reset_index()
    
    best_params = param_stats.loc[param_stats.groupby('model')[('cut_diff', 'mean')].idxmin()]
    
    print("\nBest Parameters by Model:")
    for _, row in best_params.iterrows():
        print(f"Model: {row['model']}, Param ID: {row['param_id']}")
        print(f"  Mean cut_diff: {row[('cut_diff', 'mean')]:.2f}")
        print(f"  Min cut_diff: {row[('cut_diff', 'min')]:.2f}")
        print(f"  Mean runtime: {row[('runtime', 'mean')]:.2f}s")
    
    # Create plots
    # 1. Distribution of cut_diff by model
    plt.figure(figsize=(10, 6))
    for model in results['model'].unique():
        model_data = results[results['model'] == model]['cut_diff']
        plt.hist(model_data, alpha=0.5, label=model, bins=20)
    
    plt.xlabel('Difference from Optimum Cut Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cut Value Difference by Model')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'cut_diff_by_model.png'))
    
    # 2. Runtime by model
    plt.figure(figsize=(10, 6))
    for model in results['model'].unique():
        model_data = results[results['model'] == model]['runtime']
        plt.hist(model_data, alpha=0.5, label=model, bins=20)
    
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Runtime by Model')
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, 'runtime_by_model.png'))
    
    # 3. Parameter-specific plots
    # Example: NEC model with different alpha_rate values
    if 'NEC' in results['model'].unique():
        nec_results = results[results['model'] == 'NEC']
        if 'alpha_rate' in nec_results.columns:
            plt.figure(figsize=(10, 6))
            for alpha_rate in nec_results['alpha_rate'].unique():
                data = nec_results[nec_results['alpha_rate'] == alpha_rate]['cut_diff']
                plt.hist(data, alpha=0.5, label=f'alpha_rate={alpha_rate}', bins=20)
            
            plt.xlabel('Difference from Optimum Cut Value')
            plt.ylabel('Frequency')
            plt.title('NEC Model: Effect of alpha_rate')
            plt.legend()
            plt.savefig(os.path.join(args.out_dir, 'nec_alpha_rate.png'))
    
    # Add success rate analysis
    print("\nAnalyzing Success Rates...")
    
    # Define what constitutes success - exact match or within a tolerance
    # Here we use exact match to optimum cut/energy
    if 'opt_cut' in results.columns and 'opt_energy' in results.columns:
        results['reached_optimum_cut'] = results['cut_value'] == results['opt_cut']
        results['reached_optimum_energy'] = results['energy'] == results['opt_energy']
        
    # Calculate success rates by model
    model_success = results.groupby('model').agg({
        'reached_optimum_cut': 'mean',
        'reached_optimum_energy': 'mean'
    }).reset_index()
    
    # Convert to percentages
    model_success['reached_optimum_cut'] *= 100
    model_success['reached_optimum_energy'] *= 100
    
    print("\nSuccess Rates by Model:")
    print(model_success)
    
    # Calculate success rates by model and parameter set
    param_success = results.groupby(['model', 'param_id']).agg({
        'reached_optimum_cut': 'mean',
        'reached_optimum_energy': 'mean',
        'cut_diff': 'mean',
        'seed': 'count'  # number of runs
    }).reset_index()
    
    # Convert to percentages
    param_success['reached_optimum_cut'] *= 100
    param_success['reached_optimum_energy'] *= 100
    param_success.rename(columns={'seed': 'num_runs'}, inplace=True)
    
    print("\nTop 5 Parameter Sets by Success Rate:")
    print(param_success.sort_values('reached_optimum_cut', ascending=False).head())
    
    # Create success rate bar chart
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(model_success))
    
    plt.bar(x - bar_width/2, model_success['reached_optimum_cut'], 
            width=bar_width, label='Reached Optimum Cut')
    plt.bar(x + bar_width/2, model_success['reached_optimum_energy'], 
            width=bar_width, label='Reached Optimum Energy')
    
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Model Type')
    plt.xticks(x, model_success['model'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.out_dir, 'success_rate_by_model.png'))
    
    # Create heatmap of success rates for each model's parameters
    for model in results['model'].unique():
        model_data = param_success[param_success['model'] == model]
        
        if len(model_data) > 1:  # Only create heatmap if there are multiple parameter sets
            # Extract key parameters for this model
            if model == 'POLYNOMIAL':
                if 'poly_order' in results.columns:
                    pivot_table = create_success_heatmap(
                        results[results['model'] == model], 
                        'poly_order', 'gamma_schedule', 
                        args.out_dir, f'{model.lower()}_success_heatmap.png'
                    )
            
            elif model == 'NEC':
                if 'alpha_rate' in results.columns and 'gamma' in results.columns:
                    pivot_table = create_success_heatmap(
                        results[results['model'] == model], 
                        'alpha_rate', 'gamma',
                        args.out_dir, f'{model.lower()}_success_heatmap.png'
                    )
            
            elif model == 'SIGMOID':
                if 'alpha' in results.columns:
                    # For SIGMOID, we'll look at success rate by alpha value
                    success_by_alpha = results[results['model'] == model].groupby('alpha').agg({
                        'reached_optimum_cut': 'mean',
                        'seed': 'count'
                    }).reset_index()
                    
                    success_by_alpha['reached_optimum_cut'] *= 100
                    plt.figure(figsize=(10, 6))
                    plt.bar(success_by_alpha['alpha'].astype(str), success_by_alpha['reached_optimum_cut'])
                    plt.xlabel('Alpha Value')
                    plt.ylabel('Success Rate (%)')
                    plt.title('SIGMOID Model: Success Rate by Alpha Value')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(args.out_dir, 'sigmoid_alpha_success.png'))

def create_success_heatmap(data, param1, param2, output_dir, filename):
    """Create a heatmap showing success rates for combinations of two parameters"""
    # Group by the two parameters and calculate success rate
    pivot_data = data.groupby([param1, param2])['reached_optimum_cut'].mean().reset_index()
    # Convert to percentage
    pivot_data['reached_optimum_cut'] *= 100
    
    # Create pivot table for heatmap
    pivot_table = pivot_data.pivot(index=param1, columns=param2, values='reached_optimum_cut')
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.1f', 
                cbar_kws={'label': 'Success Rate (%)'})
    plt.title(f'Success Rate by {param1} and {param2}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    
    return pivot_table

if __name__ == "__main__":
    main()
