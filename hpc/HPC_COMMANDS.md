# HPC Command Reference

This document provides a quick reference for common HPC workflow commands and custom scripts used in this project.

## Custom Scripts

### `free_cores_by_domain.sh <domain>`
Check available cores in a specific queue/domain.

```bash
free_cores_by_domain.sh hpc
free_cores_by_domain.sh fotonano
```

### `poll_free_cores_by_domain.sh <domain> [interval]`
Continuously poll and display available cores in a domain with optional refresh interval (seconds).

```bash
# Poll fotonano queue with default interval
poll_free_cores_by_domain.sh fotonano

# Poll hpc queue every 20 seconds
poll_free_cores_by_domain.sh hpc 20
```

## LSF Job Management

### Job Submission
```bash
# Submit a job script
bsub < hpc_submit_driver.sh
bsub < fotonano_submit_driver.sh
bsub < amd_submit_driver.sh
```

### Job Monitoring
```bash
# Check your job status
bstat

# Alternative job listing
bjobs

# Check queue information
bqueues

# Check specific queue details
bqueues -l hpc
bqueues -l fotonano
```

### Node Status
```bash
# Check all nodes
nodestat

# Check nodes in specific domain
nodestat hpc
nodestat fotonano

# Full node details (-F flag)
nodestat -F fotonano
```

### Job Control
```bash
# Kill a specific job
bkill 25359855

# Kill all your jobs
bkill 0
```

## Log Monitoring

### Real-time Log Tailing
```bash
cd hpc/logs/

# Follow a running job's output
tail -f driver.25359855.out

# Check error output
tail -f driver.25359855.err

# View last lines without following
tail driver.25359855.out
```

### Log Inspection
```bash
# View entire log with pagination
less driver.25359855.out

# Quick peek at end of log
tail driver.25359855.out
```

## Python Environment Setup

```bash
# Load required modules
module load python3/3.13.2
module load mpi4py/4.0.2-python-3.13.2-openmpi-5.0.6

# Activate virtual environment
source hpc/potts-env/bin/activate

# Check installed packages
pip freeze
pip freeze | grep scipy
```

## Running Sweeps

### Estimate Wall Time
```bash
# Estimate with current MPI size
python hpc/run_potts_sweep.py --config hpc/configs/250628_G1-5_cim.yaml --estimate_wall_time

# Estimate with specific number of ranks
python hpc/run_potts_sweep.py --config hpc/configs/250628_G1-5_cim.yaml --estimate_wall_time 48
python hpc/run_potts_sweep.py --config hpc/configs/250628_G1-5_cim.yaml --estimate_wall_time 72
```

### Visualize Schedules
```bash
# Generate schedule plots without running simulations
python hpc/run_potts_sweep.py --config hpc/configs/250628_G1-5_cim.yaml --plot_schedules
```

### Local Test Run
```bash
# Run locally (single process, no MPI)
python hpc/run_potts_sweep.py --config hpc/configs/local_test.yaml
```

## Results Management

### Check Results
```bash
cd hpc/results/
ls -ltr  # List by time, newest last
ls *G1*  # Find results for specific graph set
ls *rank*  # Find rank-specific results (before merging)
```

### Check Logs
```bash
cd hpc/logs/
ls -ltr  # List by time, newest last
```

### Clean Up
```bash
# Remove old config files
cd hpc/configs/
rm *max-4-cut*

# Remove old result files
cd hpc/results/
rm *poly_NEC*

# Clear logs directory
cd hpc/logs/
rm -f *
```

## Typical Workflow Sequence

### 1. Check Available Resources
```bash
cd potts_sim/
free_cores_by_domain.sh hpc
free_cores_by_domain.sh fotonano
nodestat -F fotonano
```

### 2. Estimate Job Runtime
```bash
python run_potts_sweep.py --config configs/250628_G1-5_cim.yaml --estimate_wall_time 72
```

### 3. Edit and Submit Job
```bash
nano hpc_submit_driver.sh
bsub < hpc_submit_driver.sh
```

### 4. Monitor Job
```bash
# Check job status
bstat

# Follow output in real-time
cd logs/
tail -f driver.25359855.out
```

### 5. Check Results
```bash
cd results/
ls -ltr
```

## Submit Scripts Reference

The repository uses three main submit scripts for different queues:

- **`hpc_submit_driver.sh`** - Standard HPC queue
- **`fotonano_submit_driver.sh`** - Fotonano queue (specific hardware)
- **`amd_submit_driver.sh`** - AMD queue
- **`2_hpc_submit_driver.sh`** - Alternative HPC configuration
- **`2_fotonano_submit_driver.sh`** - Alternative Fotonano configuration

All scripts follow similar structure but differ in:
- Queue name (`#BSUB -q`)
- Number of cores/nodes
- Wall time limits
- Configuration file being run

## Common Patterns

### Kill and Resubmit
```bash
bstat
bkill 25359855
nano hpc_submit_driver.sh
bsub < hpc_submit_driver.sh
```

### Monitor Multiple Jobs
```bash
# Check status
bstat

# Check each job's output
cd logs/
tail driver.25359855.out
tail driver.25359897.out
tail driver.25456581.out
```

### Wait for Resources
```bash
# Poll until resources available
poll_free_cores_by_domain.sh fotonano 120

# Then submit when ready
bsub < fotonano_submit_driver.sh
```

## Configuration File Naming Convention

Config files follow the pattern: `YYMMDD_<graph_set>_<model>_<variant>.yaml`

Examples:
- `250628_G1-5_cim.yaml` - CIM model on graphs G1-G5
- `250704_G1-5_poly_max-4-cut.yaml` - Polynomial model for max-4-cut
- `250704_G22-26_max-3-cut_mini.yaml` - Mini sweep for graphs G22-G26

## Tips

1. **Always check resources first**: Use `free_cores_by_domain.sh` before submitting
2. **Estimate before running**: Use `--estimate_wall_time` to avoid wasting resources
3. **Monitor early**: Check logs within first few minutes to catch errors early
4. **Use poll for busy queues**: `poll_free_cores_by_domain.sh` helps time submissions
5. **Kill stalled jobs quickly**: Don't waste allocation on hung processes
6. **Check results directory**: Failed jobs may leave partial rank files
7. **Clean logs regularly**: Old logs accumulate quickly
