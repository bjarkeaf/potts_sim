#!/usr/bin/env bash
###############################################################################
# free_cores.sh — per-subnet-and-arch core availability summary
#
# Usage   :  free_cores.sh <queue>
# Example :  free_cores.sh gpu-long
#
# Now uses ‘nodestat -F’ so the 5-th column contains the CPU model.  Groups
# nodes by  “<2nd-octet>-<3rd-octet>/<arch>”.
###############################################################################

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <queue>" >&2
  exit 1
fi
queue="$1"

nodestat -F "$queue" 2>/dev/null | \
awk '
  NR==1 {next}                                 # skip header line

  {
    state = $2                                 # Running, Busy, Down, etc.
    
    # Only count cores from Running or Busy nodes
    if (state != "Running" && state != "Busy") next
    
    split($1, a, "-");                         # n-62-27-11 → {"n","62","27","11"}
    subnet = a[2] "-" a[3]                     # 62-27
    arch   = $5                                # XeonE5_2660v3 (no spaces)
    group  = subnet "/" arch                   # 62-27/XeonE5_2660v3

    split($3, p, ":"); free=p[1]+0             # assumes Procs is free:total
    total = p[2]+0

    free_sum[group]     += free
    if (free > max_free[group]) max_free[group] = free

    for (t=1; t<=free; t++) nt[group, t]++
  }

  END {
    printf("\n%-28s %9s %14s %10s %12s\n",
           "Subnet/Architecture", "FreeCores", "BestTile(T)", "Nodes(T)", "C=T×Nodes")
    printf("%s\n", "--------------------------------------------------------------------------")

    for (g in free_sum) {
      bestT=0; bestC=0; nodesAtBest=0
      for (t=1; t<=max_free[g]; t++) {
        n=nt[g, t]+0; c=t*n
        if (c>bestC){bestC=c;bestT=t;nodesAtBest=n}
      }
      printf("%-28s %9d %14d %10d %12d\n",
             g, free_sum[g], bestT, nodesAtBest, bestC)
    }

    printf("\nLegend: Groups are <subnet>/<CPU-model>.  BestTile(T) maximises C.\n\n")
  }'
