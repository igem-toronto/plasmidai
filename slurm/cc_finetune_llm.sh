#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-00:00
#SBATCH --output=logs/%N-%j.out

module load StdEnv/2023 python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch pytorch_lightning wandb "pydantic<2" einops scipy pandas biopython mamba_ssm
pip install $HOME/wheels/pydantic_cli-4.3.0-py3-none-any.whl

export REPO_ROOT=~/code/plasmid-ai
cd $REPO_ROOT

# Copy source code and data
mkdir -p $SLURM_TMPDIR/plasmid-ai
cp -r src $SLURM_TMPDIR/plasmid-ai
cp -r data $SLURM_TMPDIR/plasmid-ai
cd $SLURM_TMPDIR/plasmid-ai

# Unzip plasmids
cd data
gzip -d 240212_plasmid_seq_54646.fasta.gz
mv 240212_plasmid_seq_54646.fasta plasmids.fasta
cd ..

wandb offline

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.llm.train \
    --accelerator=gpu --devices=1 \
    --precision=16-mixed \
    --batch_size=64 --num_workers=4 \
    --enable_fused_add_norm \
    --enable_wandb --wandb_dir="$REPO_ROOT/logs" \
    --enable_checkpoint --checkpoint_dir="$REPO_ROOT/checkpoints/finetune-$(date +'%M-%H-%d-%m-%Y')" \
    --enable_progress_bar \
    --max_epochs=150 --scheduler_span=5000 --train_steps_per_epoch=1000000 --val_steps_per_epoch=1000000 \
    --finetune_path="$REPO_ROOT/checkpoints/34-00-15-03-2024/last.ckpt"
