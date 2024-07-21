#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=0-15:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

export PROJECT=/lustre07/scratch/adibvafa
export REPO_ROOT=$PROJECT/plasmid-lm
cd $REPO_ROOT

module load StdEnv/2023 python/3.11 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index "torch<2.3" lightning wandb einops scipy pandas biopython transformers jsonargparse
pip install -U mamba_ssm=='2.1.0' causal_conv1d=='1.3.0.post1'

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
