# Plan for convergence test plotting script

This document describes the requirements for a script in the 'hpc' folder called 'plot_convergence.py'. 

The script generates plot of the figure of merit (FOM, specified by --fom) versus the swept parameter (specified by --conv_type) for the data (specified by --data). It generates separate plots for different models and by default separate subplots for different graphs (though --graph_grouping can be used to specify how multiple graphs should be combined in the same subplot).

See hpc/plot_benchmark.py (especially plot_rel_gap_distributions_by_graph) for example plotting code.

## Arguments

- --data (required)
  - Specifies the path to the data file to base plots on,
- --conv_type (meaning convergence type, required) 
  - Specifies which parameter is swept in the data (e.g. T or dt), which will be the x-axis of the plot.
  - Options: 
    - 'simulation_time', meaning T
    - 'time_step', meaning dt
    - Perhaps more to come
- --fom (meaning figure of merit, default: 'mean_gap')
  - Specifies which parameter to check convergence of, which will be y-axis of the plot.
  - Options:
    - 'mean_gap' (mean optimality gap across runs)
    - 'success_rate' (fraction of runs that found ground states in %)
- --graph_grouping (default: 'per_graph')
  - Specifies how to group graphs (e.g. one plot per graph, or a plot covering multiple graphs)
  - Options:
    - 'per_graph': One subplot per graph
    - 'all_graphs': One subplot averaging over all graphs
    - 'by_graph_size': Have as many subplots as number of unique graph sizes and group graphs with same graph size in one plot

## Plot type versus FOM and graph grouping

* If FOM is mean optimality gap, plots should be boxplots of optimality gap distribution (either for single graph or combined distribution for multiple graphs) over the simulation runs
* If FOM is success rate
  * For 'per_graph' subplots, use scatter plot of success rate versus swept parameter
  * For grouped ('all_graphs' or 'by_graph_size') subplots, make box plot of success rate distribution over the graphs in the group