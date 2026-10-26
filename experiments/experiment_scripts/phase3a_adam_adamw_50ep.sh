#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P3A_ADAMW_50
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=8:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase3a_adamw_50ep_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

LOSSES=(ce_dice ce_w_dice_boundary)
LOSS_KWARGS=('{}' '{"boundary_weight": 0.001}')
LOSS_NAMES=(ce_dice ce_w_dice_boundary_bw0.001)

OPTIMIZERS=(adam adamw adamw)
OPT_KWARGS=(
  '{}'
  '{"weight_decay": 0.0001}'
  '{"weight_decay": 0.001}'
)
OPT_NAMES=(adam adamw_wd1e-4 adamw_wd1e-3)

SEED=42
NUM_OPTS=${#OPTIMIZERS[@]}

LOSS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_OPTS))
OPT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_OPTS))

LOSS=${LOSSES[$LOSS_IDX]}
LOSS_KWARG=${LOSS_KWARGS[$LOSS_IDX]}
LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
OPTIMIZER_KWARGS=${OPT_KWARGS[$OPT_IDX]}
OPT_NAME=${OPT_NAMES[$OPT_IDX]}

DEST="results/segthor/phase3a_opt_${OPT_NAME}_${LOSS_NAME}_50ep_s${SEED}"

srun python main.py \
  --dataset SEGTHOR \
  --mode full \
  --loss "$LOSS" \
  --loss_kwargs "$LOSS_KWARG" \
  --net enet \
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
