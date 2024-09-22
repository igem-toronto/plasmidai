#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=1-12:00
#SBATCH --output=logs/%N-%j.out

# Add your project directory path here
export PROJECT=# <YOUR_PROJECT_DIRECTORY>

module load StdEnv/2023 python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index "torch<2.3" lightning wandb einops scipy pandas biopython transformers "mamba_ssm<2" causal_conv1d
# Add the path to your custom wheel file
pip install # <PATH_TO_JSONARGPARSE_WHEEL>

# Add the path to your repository root
export REPO_ROOT=# <YOUR_REPO_ROOT_PATH>
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