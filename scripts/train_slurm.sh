#!/usr/bin/env bash
# Adjust account/partition/time/memory for your cluster before submission.
# Create runs/slurm before sbatch so Slurm can open the log.
#SBATCH --job-name=wing-hml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the mata-src project root}"
PYTHON="${PYTHON:-python}"
METHOD="${METHOD:-hml}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/default.yaml}"
"$PYTHON" -m wing_hml train --config "$CONFIG" --method "$METHOD" --seed "$SEED" --set device=cuda --auto-resume
