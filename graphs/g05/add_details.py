#!/usr/bin/env python3
"""
Script to add maximum eigenvalues and optimum cut/energy values to g05 graph files.

This script:
1. Reads g05_solutions.tsv to get optimum max-3-cut values
2. For each .col file in the current directory:
   - Uses parse_graph from potts_utils to compute maximum eigenvalue
   - Adds optimum cut value (max-3-cut) from solutions file
   - Adds optimum energy (max-3-cut) = num_edges - optimum_cut
   - Updates the file with these comment lines
"""

import sys
import os
import re
from pathlib import Path

# Add parent directory to path to import potts_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from potts_utils import parse_graph


def read_solutions_tsv(tsv_file):
    """Read the solutions TSV file and return a dict mapping graph name to optimum cut value."""
    solutions = {}

    with open(tsv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                graph_name = parts[0]
                # Last column is the optimum cut value for max-3-cut
                opt_cut = int(parts[-1])
                solutions[graph_name] = opt_cut

    return solutions


def update_col_file(col_file, opt_cut_dict, num_states=3):
    """
    Update a .col file with optimum cut/energy and maximum eigenvalue.

    Args:
        col_file: Path to the .col file
        opt_cut_dict: Dictionary mapping graph names to optimum cut values
        num_states: Number of states for max-k-cut (default: 3 for max-3-cut)
    """
    # Get graph name from filename (e.g., g05_30.0.col -> g05_30.0)
    graph_name = Path(col_file).stem

    # Get optimum cut value from solutions
    if graph_name not in opt_cut_dict:
        print(f"Warning: No solution found for {graph_name}, skipping...")
        return False

    opt_cut = opt_cut_dict[graph_name]

    # Parse the graph to get num_edges and compute mu_max
    # parse_graph will compute and cache mu_max in the file
    num_vertices, num_edges, edges, opt_cut_from_file, opt_energy_from_file, mu_max = parse_graph(
        col_file, zero_based=False
    )

    # Calculate optimum energy
    opt_energy = num_edges - opt_cut

    # Read the current file content
    with open(col_file, 'r') as f:
        lines = f.readlines()

    # Check if file already has our comment lines
    has_opt_cut = any(f'Optimum cut value (max-{num_states}-cut)' in line for line in lines)
    has_opt_energy = any(f'Optimum energy (max-{num_states}-cut)' in line for line in lines)
    has_mu_max = any('Maximum eigenvalue' in line for line in lines)

    # Find the position of the 'p' line
    p_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('p '):
            p_line_idx = i
            break

    if p_line_idx is None:
        print(f"Error: No 'p' line found in {col_file}")
        return False

    # Build the new comment lines
    new_comments = []

    if not has_opt_cut:
        new_comments.append(f"c Optimum cut value (max-{num_states}-cut): {opt_cut}\n")

    if not has_opt_energy:
        new_comments.append(f"c Optimum energy (max-{num_states}-cut): {opt_energy}\n")

    # Add blank comment line for separation if we added new comments
    if new_comments and not (lines[p_line_idx - 1].strip() == 'c'):
        new_comments.append("c\n")

    # Insert new comments before the 'p' line
    if new_comments:
        lines = lines[:p_line_idx] + new_comments + lines[p_line_idx:]

        # Write back to file
        with open(col_file, 'w') as f:
            f.writelines(lines)

        print(f"Updated {graph_name}: opt_cut={opt_cut}, opt_energy={opt_energy}, "
              f"mu_max={mu_max:.6f}, edges={num_edges}")
        return True
    else:
        print(f"Skipped {graph_name}: already has all required comments")
        return False


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Read solutions file
    solutions_file = script_dir / 'g05_solutions.tsv'
    if not solutions_file.exists():
        print(f"Error: Solutions file not found: {solutions_file}")
        return 1

    print(f"Reading solutions from {solutions_file}")
    opt_cut_dict = read_solutions_tsv(solutions_file)
    print(f"Loaded {len(opt_cut_dict)} solutions")

    # Verify g05_30.0 has value 188
    if 'g05_30.0' in opt_cut_dict:
        print(f"\nVerification: g05_30.0 optimum cut = {opt_cut_dict['g05_30.0']}")
        assert opt_cut_dict['g05_30.0'] == 188, "Expected g05_30.0 to have optimum cut value 188"
        print("✓ Verified g05_30.0 = 188")

    # Find all .col files in the current directory
    col_files = sorted(script_dir.glob('g05_*.col'))

    # Exclude the solutions TSV file if it was converted
    col_files = [f for f in col_files if 'solution' not in f.stem.lower()]

    print(f"\nFound {len(col_files)} .col files to process\n")

    # Process each file
    updated_count = 0
    for col_file in col_files:
        if update_col_file(col_file, opt_cut_dict):
            updated_count += 1

    print(f"\n{'='*70}")
    print(f"Processing complete: {updated_count}/{len(col_files)} files updated")
    print(f"{'='*70}")

    # Verify against existing hpc files if they exist
    print("\nVerifying against existing HPC files...")
    hpc_dir = Path(__file__).parent.parent.parent / 'hpc' / 'graphs' / 'g05'

    if hpc_dir.exists():
        for hpc_file in hpc_dir.glob('g05_*.col'):
            our_file = script_dir / hpc_file.name
            if our_file.exists():
                # Parse both files
                _, _, _, opt_cut_hpc, opt_energy_hpc, _ = parse_graph(str(hpc_file))
                _, _, _, opt_cut_ours, opt_energy_ours, _ = parse_graph(str(our_file))

                if opt_cut_hpc is not None and opt_cut_ours is not None:
                    # Get the max-3-cut values
                    hpc_cut_3 = opt_cut_hpc.get(3)
                    our_cut_3 = opt_cut_ours.get(3)
                    hpc_energy_3 = opt_energy_hpc.get(3)
                    our_energy_3 = opt_energy_ours.get(3)

                    if hpc_cut_3 is not None and our_cut_3 is not None:
                        match_cut = "✓" if hpc_cut_3 == our_cut_3 else "✗"
                        match_energy = "✓" if hpc_energy_3 == our_energy_3 else "✗"
                        print(f"  {hpc_file.stem}: cut {match_cut} ({hpc_cut_3} vs {our_cut_3}), "
                              f"energy {match_energy} ({hpc_energy_3} vs {our_energy_3})")
    else:
        print(f"  HPC directory not found: {hpc_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
