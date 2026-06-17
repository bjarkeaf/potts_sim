import numpy as np
import pandas as pd
import re

def parse_graph(file_path, zero_based=False):
    """
    Parses a graph file in DIMACS format to extract:
      - num_vertices, num_edges, edges
      - optimum cut and energy (None if not found)
      - maximum eigenvalue of the coupling matrix
    """

    sources = []
    targets = []
    coupling_coeffs = []
    num_vertices = None
    num_edges = None

    opt_cut_dict = {}
    opt_energy_dict = {}
    mu_max = None

    p_line_index = None

    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("c Optimum cut value"):
                m = re.match(r"c\s+Optimum cut value\s*\(max[-]?(\d+)[-]?cut\)\s*:\s*(\d+)", line)
                if m:
                    k = int(m.group(1))
                    opt_cut_dict[k] = int(m.group(2))
                continue
            if line.startswith("c Optimum energy"):
                m = re.match(r"c\s+Optimum energy\s*\(max[-]?(\d+)[-]?cut\)\s*:\s*(\d+)", line)
                if m:
                    k = int(m.group(1))
                    opt_energy_dict[k] = int(m.group(2))
                continue
            if line.startswith("c Maximum eigenvalue"):
                mu_max = float(line.split(":")[1].strip())
                continue
            if line.startswith("p"):
                p_line_index = i
                parts = line.split()
                if len(parts) >= 3:
                    num_vertices = int(parts[1])
                    num_edges = int(parts[2])
            elif line.startswith("e"):
                parts = line.split()
                if len(parts) >= 4:
                    sources.append(int(parts[1]) if zero_based else (int(parts[1]) - 1))
                    targets.append(int(parts[2]) if zero_based else (int(parts[2]) - 1))
                    coupling_coeffs.append(-int(parts[3])) # Assuming weights used are negative of coupling coefficients

    edges = np.array([sources, targets])
    coupling_values = np.array(coupling_coeffs)

    # If mu_max wasn't found in the file, compute it
    if mu_max is None and num_vertices is not None:
        mu_max = compute_max_eigenvalue(num_vertices, sources, targets, coupling_values)
        
        # Update the file with the computed eigenvalue
        mu_max_line = f"c Maximum eigenvalue: {mu_max}\n"
        lines.insert(p_line_index, mu_max_line)
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Updated file with computed maximum eigenvalue: {mu_max}")
    
    return num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max

def compute_max_eigenvalue(num_vertices, i_indices, j_indices, J_values):
    """
    Computes the maximum (algebraic) eigenvalue of the coupling matrix.
    
    Args:
        num_vertices (int): Number of vertices in the graph
        i_indices (list/array): Source indices of edges
        j_indices (list/array): Target indices of edges
        J_values (list/array): Coupling values for the edges
        
    Returns:
        float: The maximum eigenvalue of the coupling matrix
    """
    # Import here to avoid dependency if function isn't used
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    
    # Create sparse matrix for eigenvalue computation
    data = J_values.astype(float)
    row = i_indices
    col = j_indices
    matrix = coo_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    
    # Symmetrize the matrix (make it undirected if not already)
    matrix = matrix + matrix.T
    
    # Compute the largest eigenvalue
    mu_max = eigsh(matrix, k=1, which='LA', return_eigenvectors=False)[0]
    
    return mu_max

def compute_largest_eigenvalues_and_eigenvectors(num_vertices, i_indices, j_indices, J_values, num_eigenvalues=1):
    """
    Computes the largest (algebraic) eigenvalues and their corresponding eigenvectors of the coupling matrix.
    
    Args:
        num_vertices (int): Number of vertices in the graph
        i_indices (list/array): Source indices of edges
        j_indices (list/array): Target indices of edges
        J_values (list/array): Coupling values for the edges
        num_eigenvalues (int): Number of largest eigenvalues to compute (default is 1)
        
    Returns:
        tuple: A tuple containing:
            - eigenvalues (array): The largest eigenvalues
            - eigenvectors (array): The corresponding eigenvectors
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    
    # Create sparse matrix for eigenvalue computation
    data = J_values.astype(float)
    row = i_indices
    col = j_indices
    matrix = coo_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    
    # Symmetrize the matrix (make it undirected if not already)
    matrix = matrix + matrix.T
    
    # Compute the largest k eigenvalues and their corresponding eigenvectors
    eigenvalues, eigenvectors = eigsh(matrix, k=num_eigenvalues, which='LA')

    return eigenvalues, eigenvectors


def get_best_hyperparams(results, mode='mean_gap'):
    if mode not in ('success_rate', 'mean_gap', 'min_gap'):
        raise ValueError(f"Unknown mode: {mode}. Must be 'success_rate', 'mean_gap', or 'min_gap'")

    if mode == 'success_rate':
        if 'reached_optimum_cut' not in results.columns:
            if 'opt_cut' in results.columns and 'cut_value' in results.columns:
                results = results.copy()
                results['reached_optimum_cut'] = results['cut_value'] >= results['opt_cut']
            else:
                raise ValueError("Cannot compute success rate: 'reached_optimum_cut' column missing and cannot be derived")
        metric_col = 'reached_optimum_cut'
    else:
        if 'rel_cut_gap' not in results.columns:
            results = results.copy()
            if 'cut_value' in results.columns and 'opt_cut' in results.columns:
                results['cut_gap'] = results['opt_cut'] - results['cut_value']
                results['rel_cut_gap'] = 100 * results['cut_gap'] / results['opt_cut'].replace(0, np.nan)
            elif 'cut_gap' in results.columns and 'opt_cut' in results.columns:
                results['rel_cut_gap'] = 100 * results['cut_gap'] / results['opt_cut'].replace(0, np.nan)
            else:
                raise ValueError("Cannot compute relative cut gap: required columns missing")
        metric_col = 'rel_cut_gap'

    if mode == 'success_rate':
        grouped = results.groupby(['graph', 'model', 'param_id'])[metric_col].mean().reset_index()
        grouped.rename(columns={metric_col: 'success_rate'}, inplace=True)
        best_params = grouped.loc[grouped.groupby(['graph', 'model'])['success_rate'].idxmax()]
    elif mode == 'mean_gap':
        grouped = results.groupby(['graph', 'model', 'param_id'])[metric_col].mean().reset_index()
        grouped.rename(columns={metric_col: 'mean_rel_gap'}, inplace=True)
        best_params = grouped.loc[grouped.groupby(['graph', 'model'])['mean_rel_gap'].idxmin()]
    else:
        agg_df = results.groupby(['graph', 'model', 'param_id']).agg(
            min_rel_gap=(metric_col, 'min'),
            count=(metric_col, 'size')
        ).reset_index()
        min_counts = results.groupby(['graph', 'model', 'param_id', metric_col]).size().reset_index(name='min_count')
        agg_df = pd.merge(agg_df, min_counts, on=['graph', 'model', 'param_id'], how='left')
        agg_df = agg_df[agg_df[metric_col] == agg_df['min_rel_gap']]
        agg_df = agg_df.sort_values(
            ['graph', 'model', 'min_rel_gap', 'min_count'],
            ascending=[True, True, True, False]
        )
        best_params = agg_df.drop_duplicates(subset=['graph', 'model'], keep='first')
        best_params = best_params[['graph', 'model', 'param_id', 'min_rel_gap']]

    return best_params.reset_index(drop=True)


def save_best_hyperparams_csv(results, output_path, mode='mean_gap', model_order=None):
    if 'rel_cut_gap' not in results.columns:
        results = results.copy()
        if 'cut_value' in results.columns and 'opt_cut' in results.columns:
            results['cut_gap'] = results['opt_cut'] - results['cut_value']
            results['rel_cut_gap'] = 100 * results['cut_gap'] / results['opt_cut'].replace(0, np.nan)
        elif 'cut_gap' in results.columns and 'opt_cut' in results.columns:
            results['rel_cut_gap'] = 100 * results['cut_gap'] / results['opt_cut'].replace(0, np.nan)

    if 'reached_optimum_cut' not in results.columns:
        results = results.copy()
        if 'opt_cut' in results.columns and 'cut_value' in results.columns:
            results['reached_optimum_cut'] = results['cut_value'] >= results['opt_cut']

    best_params = get_best_hyperparams(results, mode=mode)
    filtered_results = pd.merge(
        results,
        best_params[['graph', 'model', 'param_id']],
        on=['graph', 'model', 'param_id']
    )

    if model_order is None:
        model_order = sorted(filtered_results['model'].unique())

    potential_params = [
        'poly_order', 'gamma_factor', 'beta_factor',
        'alpha_rate', 'r_target', 'alpha',
        'B_num_vertices', 'zeta', 'gamma_rate/gamma_th'
    ]
    model_hyperparams = {}
    for model in model_order:
        model_data = results[results['model'] == model]
        if not model_data.empty:
            swept = [p for p in potential_params
                     if p in model_data.columns and len(model_data[p].unique()) > 1]
            model_hyperparams[model] = swept

    table_rows = []
    for model in model_order:
        model_results = filtered_results[filtered_results['model'] == model]
        if model_results.empty:
            continue
        graphs = sorted(model_results['graph'].unique())
        swept_params = model_hyperparams.get(model, [])
        for graph in graphs:
            graph_data = model_results[model_results['graph'] == graph]
            if graph_data.empty:
                continue
            param_values = []
            for param in swept_params:
                if param in graph_data.columns:
                    unique_vals = graph_data[param].unique()
                    if len(unique_vals) == 1:
                        val = unique_vals[0]
                        if isinstance(val, float):
                            val = f"{val:.2f}"
                        else:
                            val = str(val)
                        param_values.append(f"{param}={val}")
            row = {
                'Model': model,
                'Graph': graph,
                'param_id': graph_data['param_id'].iloc[0],
                'Hyperparameter 1': param_values[0] if len(param_values) > 0 else '',
                'Hyperparameter 2': param_values[1] if len(param_values) > 1 else '',
                'Mean rel. gap (%)': f"{graph_data['rel_cut_gap'].mean():.2f}",
                'Median rel. gap (%)': f"{graph_data['rel_cut_gap'].median():.2f}",
                'Min rel. gap (%)': f"{graph_data['rel_cut_gap'].min():.2f}",
                'Std rel. gap (%)': f"{graph_data['rel_cut_gap'].std():.2f}",
            }
            if 'reached_optimum_cut' in graph_data.columns:
                row['Success rate (%)'] = f"{100 * graph_data['reached_optimum_cut'].mean():.2f}"
            table_rows.append(row)

    stats_df = pd.DataFrame(table_rows)
    if not stats_df.empty:
        stats_df['Model'] = pd.Categorical(stats_df['Model'], categories=model_order, ordered=True)
        stats_df = stats_df.sort_values(['Model', 'Graph'])

    stats_df.to_csv(output_path, index=False)
    print(f"Saved best hyperparameters table ({mode} mode) to {output_path}")
    return stats_df