#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=P1C_BOUNDARY
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase1c_boundary_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

LOSSES=(ce_sqrt_w_dice ce_w_dice_boundary ce_w_dice_boundary ce_sqrt_w_dice_boundary ce_sqrt_w_dice_boundary)
LOSS_KWARGS=('{}' '{"boundary_weight": 0.01}' '{"boundary_weight": 0.001}' '{"boundary_weight": 0.01}' '{"boundary_weight": 0.001}')
NAMES=(ce_sqrt_w_dice ce_w_dice_boundary_bw0.01 ce_w_dice_boundary_bw0.001 ce_sqrt_w_dice_boundary_bw0.01 ce_sqrt_w_dice_boundary_bw0.001)

LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}
LOSS_KWARG=${LOSS_KWARGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

SEED=42
DEST="results/segthor/phase1c_boundary_${NAME}_s${SEED}"

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
