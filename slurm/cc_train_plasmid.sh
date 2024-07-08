#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=1-12:00
#SBATCH --output=logs/%N-%j.out

export PROJECT=~/projects/def-mikeuoft/alstonlo

module load StdEnv/2023 python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index "torch<2.3" lightning wandb einops scipy pandas biopython transformers "mamba_ssm<2" causal_conv1d
pip install $PROJECT/wheels/jsonargparse-4.31.0-py3-none-any.whl

export REPO_ROOT=$PROJECT/code/plasmid-lm
cd $REPO_ROOT

wandb offline

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.train \
    --backend.matmul_precision=medium \
    --data.batch_size=64 --data.num_workers=4 \
    --lit.fused_add_norm=true --lit.scheduler_span=50000 --lit.top_p=0.9 \
    --trainer.accelerator=gpu  --trainer.devices=2 --trainer.precision=bf16-mixed \
    --trainer.wandb=true --trainer.wandb_dir="${REPO_ROOT}/logs" \
    --trainer.checkpoint=true --trainer.checkpoint_dir="${REPO_ROOT}/checkpoints/$(date +'%Y-%m-%d-%H-%M')" \
    --trainer.progress_bar=true \
    --trainer.max_epochs=175