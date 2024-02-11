#!/bin/bash

#SBATCH --job-name=train_edm
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

source ~/.bashrc
conda activate plasmid-ai

cd ..

srun python -m src.experimental.llm.train \
  --accelerator=gpu \
  --devices=2 \
  --precision=bf16-mixed \
  --batch_size=32 \
  --num_workers=4 \
  --enable_fused_add_norm \
  --enable_wandb \
  --enable_checkpoint --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}