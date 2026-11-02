#!/bin/bash
#SBATCH --job-name=malaria-splits-1m
#SBATCH --output=sbatch/logs-sim/splits_1m_%A_%a.out
#SBATCH --error=sbatch/logs-sim/splits_1m_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2%1
#SBATCH --chdir=/nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim

# Submit from the repository root after creating sbatch/logs-sim.
# Independent seeds/stores keep whole simulation batches within one split.
# The test store is reserved for final evaluation, not training or tuning.
set -euo pipefail
cd /nfs/jbailey5/baileyweb/cmeiersc/Projects/transmission-sim

case "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}" in
  0) split=train;      batches=7813; seed=2026103121 ;;
  1) split=validation; batches=782;  seed=2026103122 ;;
  2) split=test;       batches=782;  seed=2026103123 ;;
  *) echo "Unexpected array task ID" >&2; exit 1 ;;
esac

printf 'Generating split=%s trajectories=%s seed=%s commit=%s\n' \
  "$split" "$((batches * 128))" "$seed" "$(git rev-parse HEAD)"

uv run --frozen --extra cuda12 python -u -m examples.build_malaria_dataset \
  seed="$seed" \
  num_generations=120 \
  num_batches="$batches" \
  num_simulations_per_batch=128 \
  output.num_batches_per_write=8 \
  output.overwrite=false \
  output.store="outputs/transmission_120gen_1m_v2/${split}.zarr" \
  hydra.run.dir="outputs/transmission_120gen_1m_v2/hydra/${split}"
