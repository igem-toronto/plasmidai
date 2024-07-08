#!/bin/bash

#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

source ~/.bashrc
conda activate plasmid-lm

cd ..

srun python -m src.experimental.llm.train \
  --accelerator=gpu \
  --devices=1 \
  --precision=16-mixed \
  --batch_size=32 \
  --num_workers=8 \
  --enable_fused_add_norm \
  --enable_wandb \
  --enable_checkpoint --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}