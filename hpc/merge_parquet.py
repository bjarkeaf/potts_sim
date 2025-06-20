#!/usr/bin/env python3
# filepath: /home/bjarke/Documents/PM/potts_sim/hpc/merge_parquet.py
import pandas as pd
import argparse
import os
from pathlib import Path

# How to use:
# python merge_parquet.py --input file1.parquet file2.parquet [file3.parquet ...] [--output merged.parquet]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge multiple parquet files into one')
    parser.add_argument('--input', nargs='+', required=True, help='Input parquet files to merge')
    parser.add_argument('--output', type=str, help='Output parquet file name (optional)')
    args = parser.parse_args()
    
    # Validate input files
    for file_path in args.input:
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} does not exist")
            return 1
        if not file_path.endswith('.parquet'):
            print(f"Warning: Input file {file_path} does not have .parquet extension")
    
    # Generate output file name if not provided
    if args.output is None:
        # Look for config names in the results_X.parquet pattern
        config_names = []
        for file_path in args.input:
            name = Path(file_path).stem
            if name.startswith("results_"):
                config_names.append(name[8:])  # Skip "results_" prefix
            else:
                config_names.append(name)
        
        # Create output filename
        if len(config_names) <= 3:
            # For few files, include all config names
            args.output = f"results_{'_'.join(config_names)}.parquet"
        else:
            # For many files, use a more generic name
            args.output = f"results_merged_{len(args.input)}_files.parquet"
    
    # Read and merge parquet files
    dfs = []
    total_rows = 0
    
    for file_path in args.input:
        try:
            df = pd.read_parquet(file_path)
            dfs.append(df)
            print(f"Read {len(df)} rows from {file_path}")
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 1
    
    if not dfs:
        print("Error: No valid input files to merge")
        return 1
    
    # Concatenate DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged DataFrame to output file
    try:
        merged_df.to_parquet(args.output, index=False)
        print(f"Successfully merged {len(merged_df)} rows to {args.output}")
        
        # Verify row count
        if len(merged_df) != total_rows:
            print(f"Warning: Expected {total_rows} rows but got {len(merged_df)} rows in merged file")
    except Exception as e:
        print(f"Error writing to {args.output}: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())