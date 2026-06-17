#!/usr/bin/env bash
set -e

python dynamics_figure.py

python plot_benchmark.py --results results/paper/results_260123_gset_max-3-cut.parquet --figure_mode
python plot_benchmark.py --results results/paper/results_260123_gset_max-4-cut.parquet --figure_mode
python plot_benchmark.py --results results/paper/results_260123_g05.parquet --figure_mode

python plot_convergence.py --data results/paper/results_260128_gset_max-3-cut_convergence_sim_time.parquet --conv_type simulation_time --figure_mode
python plot_convergence.py --data results/paper/results_260128_gset_max-3-cut_convergence_time_step.parquet --conv_type time_step --figure_mode
python plot_convergence.py --data results/paper/results_260128_gset_max-4-cut_convergence_sim_time.parquet --conv_type simulation_time --figure_mode
python plot_convergence.py --data results/paper/results_260128_gset_max-4-cut_convergence_time_step.parquet --conv_type time_step --figure_mode
python plot_convergence.py --data results/paper/results_260128_g05_convergence_sim_time.parquet --conv_type simulation_time --figure_mode
python plot_convergence.py --data results/paper/results_260128_g05_convergence_time_step.parquet --conv_type time_step --figure_mode
