def parse_graph(file_path, zero_based=False):
    """
    Parses a graph file in DIMACS format to extract:
      - num_spins, num_edges, edges
      - optimum cut and energy (None if not found)
      - maximum eigenvalue of the coupling matrix
    """
    import numpy as np

    sources = []
    targets = []
    coupling_coeffs = []
    num_spins = None
    num_edges = None

    opt_cut = None
    opt_energy = None
    mu_max = None

    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c Optimum cut value"):
                opt_cut = int(line.split(":")[1].strip())
                continue
            if line.startswith("c Optimum energy"):
                opt_energy = int(line.split(":")[1].strip())
                continue
            if line.startswith("c Maximum eigenvalue"):
                mu_max = float(line.split(":")[1].strip())
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 3:
                    num_spins = int(parts[1])
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
    if mu_max is None and num_spins is not None:
        mu_max = compute_max_eigenvalue(num_spins, sources, targets, coupling_values)
        
        # Update the file with the computed eigenvalue
        mu_max_line = f"c Maximum eigenvalue: {mu_max}\n"
        p_line_index = next((i for i, line in enumerate(lines) if line.strip().startswith("p")), 0)
        lines.insert(p_line_index, mu_max_line)
        
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        print(f"Updated file with computed maximum eigenvalue: {mu_max}")
    
    return num_spins, num_edges, edges, opt_cut, opt_energy, mu_max

def compute_max_eigenvalue(num_spins, i_indices, j_indices, J_values):
    """
    Computes the maximum eigenvalue of the coupling matrix.
    
    Args:
        num_spins (int): Number of spins in the graph
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
    matrix = coo_matrix((data, (row, col)), shape=(num_spins, num_spins))
    
    # Symmetrize the matrix (make it undirected if not already)
    matrix = matrix + matrix.T
    
    # Compute the largest eigenvalue
    mu_max = eigsh(matrix, k=1, which='LA', return_eigenvectors=False)[0]
    
    return mu_max
