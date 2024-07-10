#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=1-12:00
#SBATCH --output=logs/%N-%j.out

export REPO_ROOT=~/projects/def-mikeuoft/alstonlo/code/plasmid-lm
cd $REPO_ROOT

bash slurm/cc_deps.sh

wandb offline

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.train \
    --backend.matmul_precision=medium \
    --data RepliconDataModule --data.max_tokens=262144 --data.batch_size=160 --data.num_workers=8 \
    --lit.hidden_features=256 --lit.num_layers=20 \
    --lit.fused_add_norm=true --lit.scheduler_span=20000 --lit.num_samples_per_epoch=20 --lit.top_p=0.9 \
    --trainer.accelerator=gpu  --trainer.devices=1 --trainer.precision=bf16-mixed \
    --trainer.wandb=true --trainer.wandb_project=train_replicon_llm --trainer.wandb_dir="${REPO_ROOT}/logs" \
    --trainer.checkpoint=true --trainer.checkpoint_dir="${REPO_ROOT}/checkpoints/$(date +'%Y-%m-%d-%H-%M')" \
    --trainer.progress_bar=true \
    --trainer.max_epochs=100 --trainer.train_steps_per_epoch=500