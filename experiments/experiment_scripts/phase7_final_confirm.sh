#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P7_FINAL
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase7_final_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# Confirm Phase 6b 2D strong and Phase 6c 3-slice strong, stable CE / all-pair Dice.
# Copy their seed-42 references into the matching final_* folders separately.
LOSS="ce_dice"
OPTIMIZER="adam"
OPTIMIZER_KWARGS='{}'
AUGMENTATION="strong"
NETS=(enet enet25d)
SLICES_BY_NET=("" 3)
CONFIG_NAMES=(enet_strong_aug_ce_dice enet25d_3s_step1_strong_aug_ce_dice)
SEEDS=(123 2026)
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
[[ "$TASK_ID" =~ ^[0-3]$ ]] || { echo "Expected task index 0-3" >&2; exit 1; }
NET_IDX=$((TASK_ID / ${#SEEDS[@]}))
SEED_IDX=$((TASK_ID % ${#SEEDS[@]}))
NET=${NETS[$NET_IDX]}
SLICES=${SLICES_BY_NET[$NET_IDX]}
CONFIG_NAME=${CONFIG_NAMES[$NET_IDX]}
SEED=${SEEDS[$SEED_IDX]}
SLICE_STEP=1
# Match Phase 6's heart-largest / aorta-0.25-mL default.
unset POSTPROCESS_ARGS

DEST="results/segthor_new/final_${CONFIG_NAME}_25ep_s${SEED}"
[[ ! -e "$DEST" ]] || { echo "Refusing to overwrite $DEST" >&2; exit 1; }

SLICE_ARGS=()
if [[ "$NET" == "enet25d" ]]; then
  SLICE_ARGS=(--slices "$SLICES" --slice-step "$SLICE_STEP")
fi

srun python main.py \
  --dataset SEGTHOR --data-root data/SEGTHOR_new \
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
