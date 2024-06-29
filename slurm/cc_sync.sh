#!/bin/bash

export PROJECT=~/projects/def-mikeuoft/alstonlo
source $PROJECT/envs/wandb-env/bin/activate
export REPO_ROOT=$PROJECT/code/plasmid-ai
cd $REPO_ROOT
wandb sync logs/wandb/offline-*
