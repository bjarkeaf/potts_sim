# Success rate benchmarking plan

This document describes a plan for updating hpc/plot_benchmark.py to support plots that compare success rates across different models and graphs.

## Arguments

Currently the argument --plot_type does have option 'success_rate', but the plotting code should be updated to also support the following argument:

--graph_grouping (default: 'by_graph_size')

- Specifies how to group graphs (e.g. one plot per graph, or a plot covering multiple graphs)
- Options:
  - 'per_graph': One subplot per graph
  - 'all_graphs': One subplot averaging over all graphs
  - 'by_graph_size': Have as many subplots as number of unique graph sizes and group graphs with same graph size in one plot

## Plot type versus graph grouping

* For 'per_graph' subplots, use scatter plot of success rate versus swept parameter
* For grouped ('all_graphs' or 'by_graph_size') subplots, make box plot of success rate distribution over the graphs in the group