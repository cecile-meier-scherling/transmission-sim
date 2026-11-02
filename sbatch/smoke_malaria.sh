#!/bin/bash
#SBATCH --job-name=malaria-smoke
#SBATCH --output=sbatch/logs-sim/smoke_%j.out
#SBATCH --error=sbatch/logs-sim/smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim

# Submit from the repository root after creating sbatch/logs-sim.
set -euo pipefail

cd /nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim

uv run --frozen --extra cuda12 pytest -q \
  tests/test_malaria.py tests/test_malaria_analysis.py

# Run only if pytest succeeds; refuse to overwrite an existing dataset.
uv run --frozen --extra cuda12 python -m examples.build_malaria_dataset \
  seed=20261031 \
  num_generations=120 \
  num_batches=2 \
  num_simulations_per_batch=16 \
  output.num_batches_per_write=1 \
  output.overwrite=false \
  output.store=outputs/transmission_smoke_10y.zarr
