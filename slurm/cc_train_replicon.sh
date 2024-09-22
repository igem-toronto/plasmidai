#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=0-15:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

# Add your project directory path here
export PROJECT=# <YOUR_PROJECT_DIRECTORY>
# Add the path to your repository root
export REPO_ROOT=# <YOUR_REPO_ROOT_PATH>
cd $REPO_ROOT

# Add the path to your conda/mamba environment
source # <PATH_TO_YOUR_MAMBA_OR_CONDA>/bin/activate

wandb offline

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.train \
    --backend.matmul_precision=medium \
    --data RepliconDataModule --data.max_tokens=131072 --data.batch_size=80 --data.num_workers=4 \
    --lit.fused_add_norm=true --lit.scheduler_span=50000 \
    --lit.num_samples_per_epoch=20 --lit.top_p=0.9 \
    --trainer.accelerator=gpu --trainer.devices=2 --trainer.precision=bf16-mixed \
    --trainer.wandb=true --trainer.wandb_project=train_replicon_llm --trainer.wandb_dir="${REPO_ROOT}/logs" \
    --trainer.checkpoint=true --trainer.checkpoint_dir="${REPO_ROOT}/checkpoints/$(date +'%Y-%m-%d-%H-%M')" \
    --trainer.progress_bar=true \
    --trainer.max_epochs=100 --trainer.train_steps_per_epoch=500