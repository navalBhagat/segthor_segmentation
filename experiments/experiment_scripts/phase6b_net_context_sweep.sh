#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P6B_CONTEXT
#SBATCH --array=0-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase6b_context_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# Stable logit CE with historical all-pair Dice averaging.
# Match Phase 6 data, seed, epochs, augmentation, optimizer and postprocessing.
LOSS="ce_dice"
OPTIMIZER="adam"
OPTIMIZER_KWARGS='{}'
AUGMENTATION="strong"
# Use the helper's heart-largest / aorta-0.25-mL default.
unset POSTPROCESS_ARGS

NETS=(enet)
SLICES=("")
SLICE_STEPS=(1)
NAMES=(enet_1slice_25ep)

NET=${NETS[$SLURM_ARRAY_TASK_ID]}
SLICE_COUNT=${SLICES[$SLURM_ARRAY_TASK_ID]}
SLICE_STEP=${SLICE_STEPS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
SEED=42
DEST="results/segthor_new/phase6b_context_${NAME}_${LOSS}_s${SEED}"

if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing run: $DEST" >&2
  exit 1
fi

SLICE_ARGS=()
if [[ "$NET" == "enet25d" ]]; then
  SLICE_ARGS=(--slices "$SLICE_COUNT" --slice-step "$SLICE_STEP")
fi

srun python main.py \
  --dataset SEGTHOR_new \
  --mode full \
  --loss "$LOSS" \
  --net "$NET" \
  ${SLICE_ARGS[@]+"${SLICE_ARGS[@]}"} \
  --augmentation "$AUGMENTATION" \
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
