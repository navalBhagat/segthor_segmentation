#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P3B_SGD_RMS
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase3b_sgd_rmsprop_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

# Schedule alongside Phase 3a with the same losses, seed, and epoch budget.
LOSSES=(ce_dice ce_w_dice_boundary)
LOSS_KWARGS=('{}' '{"boundary_weight": 0.001}')
LOSS_NAMES=(ce_dice ce_w_dice_boundary_bw0.001)
EPOCHS=50

OPTIMIZERS=(sgd rmsprop)
OPT_KWARGS=(
  '{"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001}'
  '{"lr": 0.0005, "weight_decay": 0.0001}'
)
OPT_NAMES=(sgd_lr0.01_mom_wd1e-4 rmsprop_lr0.0005_wd1e-4)

NUM_OPTS=${#OPTIMIZERS[@]}
LOSS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_OPTS))
OPT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_OPTS))

LOSS=${LOSSES[$LOSS_IDX]}
LOSS_KWARG=${LOSS_KWARGS[$LOSS_IDX]}
LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}
OPTIMIZER_KWARGS=${OPT_KWARGS[$OPT_IDX]}
OPT_NAME=${OPT_NAMES[$OPT_IDX]}
SEED=42

DEST="results/segthor/phase3b_opt_${OPT_NAME}_${LOSS_NAME}_${EPOCHS}ep_s${SEED}"

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
  --epochs "$EPOCHS" \
  --dest "$DEST" \
  --gpu \
  --val_3d \
  --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
