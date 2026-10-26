#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P5B_AUG_COMPONENTS
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase5b_augmentation_components_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# Remove one component at a time; reuse phase5_aug_strong_ce_dice_s42 as reference.
LOSS="ce_dice"
OPTIMIZER="adam"
OPTIMIZER_KWARGS='{}'
AUGMENTATION=strong

COMPONENT_ARGS=(
  "gamma noise contrast blur"
  "affine noise contrast blur"
  "affine gamma contrast blur"
  "affine gamma noise blur"
  "affine gamma noise contrast"
)
NAMES=(without_affine without_gamma without_noise without_contrast without_blur)

read -r -a COMPONENTS <<< "${COMPONENT_ARGS[$SLURM_ARRAY_TASK_ID]}"
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
SEED=42
DEST="results/segthor_new/phase5b_aug_${AUGMENTATION}_${NAME}_${LOSS}_s${SEED}"

srun python main.py \
  --dataset SEGTHOR_new \
  --mode full \
  --loss "$LOSS" \
  --net enet \
  --augmentation "$AUGMENTATION" \
  --augmentation-components "${COMPONENTS[@]}" \
  --optimizer "$OPTIMIZER" \
  --optimizer_kwargs "$OPTIMIZER_KWARGS" \
  --seed "$SEED" \
  --num_workers 8 \
  --epochs 25 \
  --dest "$DEST" \
  --gpu \
  --val_3d \
  --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
