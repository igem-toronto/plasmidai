#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=0-08:00
#SBATCH --output=logs/%N-%j.out

module load StdEnv/2023 python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index "torch<2.3" pytorch_lightning wandb "pydantic<2" einops scipy pandas biopython mamba_ssm causal_conv1d
pip install $HOME/wheels/pydantic_cli-4.3.0-py3-none-any.whl

export REPO_ROOT=~/code/plasmid-ai
cd $REPO_ROOT

wandb offline

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.llm.sample \
    --checkpoint_path="${REPO_ROOT}/checkpoints/finetune-26-22-29-05-2024/last.ckpt" \
    --accelerator=gpu  \
    --precision=bf16-mixed \
    --batch_size=100 \
    --samples_path="${REPO_ROOT}/samples/samples-$(date +'%M-%H-%d-%m-%Y').fasta"
