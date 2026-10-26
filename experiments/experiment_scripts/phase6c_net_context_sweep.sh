#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P6C_CONTEXT
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase6c_context_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

set -euo pipefail

# Tasks 0/1: 3-/5-slice context at seed 42, strong augmentation.
# Tasks 2-4: isotropic baseline-1-style ENet, CE, no augmentation, three seeds.
# Tasks 5-7: isotropic ENet, CE/Dice, no augmentation, three seeds.
# All use the same prepared data, stable CE, 25 epochs and postprocessing.
# CE/Dice uses the reverted historical all-pair Dice averaging.
OPTIMIZER="adam"
OPTIMIZER_KWARGS='{}'
# Use the helper's heart-largest / aorta-0.25-mL default.
unset POSTPROCESS_ARGS

NETS=(enet25d enet25d enet enet enet enet enet enet)
SLICES=(3 5 "" "" "" "" "" "")
SLICE_STEPS=(1 1 1 1 1 1 1 1)
NAMES=(enet25d_3s_step1_25ep enet25d_5s_step1_25ep
       baseline1_isotropic_25ep baseline1_isotropic_25ep baseline1_isotropic_25ep
       enet_1slice_none_25ep enet_1slice_none_25ep enet_1slice_none_25ep)
LOSSES=(ce_dice ce_dice ce ce ce ce_dice ce_dice ce_dice)
AUGMENTATIONS=(strong strong none none none none none none)
SEEDS=(42 42 42 123 2026 42 123 2026)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#NETS[@]}" ]]; then
  echo "Invalid task index: $TASK_ID (expected 0-7)" >&2
  exit 1
fi
NET=${NETS[$TASK_ID]}
SLICE_COUNT=${SLICES[$TASK_ID]}
SLICE_STEP=${SLICE_STEPS[$TASK_ID]}
NAME=${NAMES[$TASK_ID]}
LOSS=${LOSSES[$TASK_ID]}
AUGMENTATION=${AUGMENTATIONS[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}
DEST="results/segthor_new/phase6c_context_${NAME}_${LOSS}_s${SEED}"

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
