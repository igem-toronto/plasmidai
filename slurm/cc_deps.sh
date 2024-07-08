#!/bin/bash

export PROJECT=~/projects/def-mikeuoft/alstonlo

module load StdEnv/2023 python/3.10 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index "torch<2.3" lightning wandb einops scipy pandas biopython transformers "mamba_ssm<2" causal_conv1d
pip install $PROJECT/wheels/jsonargparse-4.31.0-py3-none-any.whl
