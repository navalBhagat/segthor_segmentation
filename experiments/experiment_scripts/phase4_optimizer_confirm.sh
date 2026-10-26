#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P4_OPT_CONFIRM
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase4_optimizer_confirm_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# Reuse Phase 3a Adam + ce_dice seed 42; confirm only the remaining seeds.
# Screen corrected sqrt-weighted Dice on one seed before broader confirmation.
LOSSES=(ce_dice ce_dice ce_sqrt_w_dice)
NAMES=(adam_50ep adam_50ep adam_gdice_v2_50ep)
SEEDS=(123 2026 42)

LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DEST="results/segthor/phase4_opt_${NAME}_${LOSS}_s${SEED}"

srun python main.py \
  --dataset SEGTHOR \
  --mode full \
  --loss "$LOSS" \
  --net enet \
  --optimizer adam \
  --optimizer_kwargs '{}' \
  --seed "$SEED" \
  --num_workers 8 \
  --epochs 50 \
  --dest "$DEST" \
  --gpu \
  --val_3d \
  --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
