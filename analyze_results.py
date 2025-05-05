#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

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
    
    print(f"\nPlots saved to {args.out_dir}")

if __name__ == "__main__":
    main()
