#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=P1_LOSS
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm/segthor/final_experiments/phase1_loss_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=navaltbhagat@gmail.com

module purge
module load 2025
module load Anaconda3/2025.06-1

cd $HOME/MedicalAIImagingGroup14/
source activate ai4mi

LOSSES=(ce dice w_dice ce_dice ce_w_dice)
LOSS=${LOSSES[$SLURM_ARRAY_TASK_ID]}

SEED=42
DEST="results/segthor/phase1_loss_${LOSS}_s${SEED}"

# srun python main.py \
#   --dataset SEGTHOR \
#   --mode full \
#   --loss "$LOSS" \
#   --net enet \
#   --optimizer adam \
#   --seed "$SEED" \
#   --num_workers 8 \
#   --epochs 25 \
#   --dest "$DEST" \
#   --gpu \
#   --val_3d \
#   --val_3d_every 1

srun bash jobs/final_experiments/postprocess_metrics_plots.sh "$DEST"
