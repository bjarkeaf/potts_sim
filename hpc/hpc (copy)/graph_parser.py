def parse_graph(file_path, zero_based=False):
    """
    Parses a graph file in DIMACS format to extract:
      - num_spins, num_edges, edges
      - optimum cut and energy (None if not found)
    """
    import numpy as np

    sources = []
    targets = []
    coupling_strengths = []
    num_spins = None
    num_edges = None

    opt_cut = None
    opt_energy = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c Optimum cut value"):
                opt_cut = int(line.split(":")[1])
                continue
            if line.startswith("c Optimum energy"):
                opt_energy = int(line.split(":")[1])
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
                    coupling_strengths.append(int(parts[3]))

    edges = np.array([sources, targets])
    return num_spins, num_edges, edges, opt_cut, opt_energy
