#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0-03:00
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

srun python -m src.experimental.sample \
    --backend.matmul_precision=medium \
    --sample.checkpoint_path="${REPO_ROOT}/checkpoints/57-19-30-06-2024/last.ckpt" \
    --sample.precision=bfloat16 --sample.num_samples=10000 --sample.top_p=0.9 \
    --sample.wandb_dir="${REPO_ROOT}/logs"