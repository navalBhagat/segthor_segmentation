#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=P2_LOSS_CONFIRM
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase2_loss_confirm_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

LOSSES=(ce_dice ce_w_dice_boundary)
LOSS_KWARGS=('{}' '{"boundary_weight": 0.001}')
NAMES=(ce_dice ce_w_dice_boundary_bw0.001)
SEEDS=(123 2026)

LOSS_IDX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))

LOSS=${LOSSES[$LOSS_IDX]}
LOSS_KWARG=${LOSS_KWARGS[$LOSS_IDX]}
NAME=${NAMES[$LOSS_IDX]}
SEED=${SEEDS[$SEED_IDX]}
DEST="results/segthor/phase2_loss_${NAME}_s${SEED}"

srun python main.py \
  --dataset SEGTHOR \
  --mode full \
  --loss "$LOSS" \
  --loss_kwargs "$LOSS_KWARG" \
  --net enet \
  --optimizer adam \
  --seed "$SEED" \
  --num_workers 8 \
  --epochs 25 \
  --dest "$DEST" \
  --gpu \
  --val_3d \
  --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
