#!/bin/bash

module load StdEnv/2023 python/3.10 scipy-stack
export PROJECT=~/projects/def-mikeuoft/alstonlo
source $PROJECT/envs/wandb-env/bin/activate
export REPO_ROOT=$PROJECT/code/plasmid-lm
cd $REPO_ROOT
watch -n 300 wandb sync $REPO_ROOT/logs/wandb/offline-*
