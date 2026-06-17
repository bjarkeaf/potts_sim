#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from potts_utils import save_best_hyperparams_csv

def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV tables with best hyperparameters from benchmark results.',
        epilog='''
Examples:
  # Using mean gap mode (default)
  python save_best_hyperparams.py results_max_3_cut.parquet
  
  # Using success rate mode
  python save_best_hyperparams.py results_max_3_cut.parquet --mode success_rate
  
  # Using min gap mode with custom output
  python save_best_hyperparams.py results_max_3_cut.parquet --mode min_gap --output my_output.csv
  
  # With custom model order
  python save_best_hyperparams.py results_max_3_cut.parquet --mode mean_gap --model-order NEC q-PDC "Polynomial PM"
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input parquet file (e.g., results_max_3_cut.parquet)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['success_rate', 'mean_gap', 'min_gap'],
        default='mean_gap',
        help='Mode for selecting best hyperparameters (default: mean_gap)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: best_params_{mode}.csv)'
    )
    parser.add_argument(
        '--model-order',
        nargs='+',
        default=None,
        help='Custom model order (e.g., --model-order NEC q-PDC "Polynomial PM")'
    )
    
    args = parser.parse_args()
    
    # Set default output filename based on mode if not specified
    if args.output:
        output_file = args.output
    else:
        # Extract input filename without extension
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_dir = 'best_hyperparams'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{input_basename}_{args.mode}.csv')
    
    # Load results
    print(f"Loading results from {args.input_file}...")
    results = pd.read_parquet(args.input_file)
    
    # Generate best hyperparameters CSV
    df = save_best_hyperparams_csv(
        results, 
        output_file, 
        mode=args.mode,
        model_order=args.model_order
    )
    
    print(f"\nFile generated successfully: {output_file}")

if __name__ == '__main__':
    main()
