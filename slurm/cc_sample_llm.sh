#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=0-04:00
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

export TORCH_NCCL_BLOCKING_WAIT=1

srun python -m src.experimental.llm.sample \
    --checkpoint_path="$REPO_ROOT/checkpoints/44-02-18-03-2024/last.ckpt" \
    --accelerator=gpu  \
    --precision=16-mixed \
    --batch_size=100 \
    --samples_path="$REPO_ROOT/samples/samples-$(date +'%M-%H-%d-%m-%Y').fasta"
