#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P5C_STRONG_CONFIRM
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor_new/final_experiments/phase5c_strong_confirm_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/MedicalAIImagingGroup14/"
source activate ai4mi

set -euo pipefail

# Reuse Phase 5 strong seed 42; store the remaining seeds alongside it.
SEEDS=(42 123 2026)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DEST="results/segthor_new/phase5_aug_strong_ce_dice_s${SEED}"

# Match the earlier Phase 5 postprocessed results, overriding the helper default.
export POSTPROCESS_ARGS="--largest --fill_holes"

srun python main.py \
  --dataset SEGTHOR_new \
  --mode full \
  --loss ce_dice \
  --net enet \
  --augmentation strong \
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
