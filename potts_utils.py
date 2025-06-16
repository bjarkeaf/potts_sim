import numpy as np
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
    ave_abs_J = None

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
            if line.startswith("c Average absolute coupling"):
                ave_abs_J = float(line.split(":")[1].strip())
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
    
    # If average absolute coupling wasn't found, compute it
    if ave_abs_J is None and coupling_values.size > 0:
        ave_abs_J = np.mean(np.abs(coupling_values))
        
        # Update the file with the computed average absolute coupling
        ave_abs_J_line = f"c Average absolute coupling: {ave_abs_J}\n"
        lines.insert(p_line_index, ave_abs_J_line)
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Updated file with computed average absolute coupling: {ave_abs_J}")

    return num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max, ave_abs_J

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