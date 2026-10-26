#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P5_AUG
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase5_augmentation_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# The none reference is copied from Phase 4 Adam + ce_dice, 50 epochs, seed 42.
LOSS="ce_dice"
OPTIMIZER="adam"
OPTIMIZER_KWARGS='{}'

AUGMENTATIONS=(mild strong)
AUGMENTATION=${AUGMENTATIONS[$SLURM_ARRAY_TASK_ID]}
SEED=42
DEST="results/segthor/phase5_aug_${AUGMENTATION}_${LOSS}_s${SEED}"

srun python main.py \
  --dataset SEGTHOR \
  --mode full \
  --loss "$LOSS" \
  --net enet \
  --augmentation "$AUGMENTATION" \
  --optimizer "$OPTIMIZER" \
  --optimizer_kwargs "$OPTIMIZER_KWARGS" \
  --seed "$SEED" \
  --num_workers 8 \
  --epochs 50 \
  --dest "$DEST" \
  --gpu \
  --val_3d \
  --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
