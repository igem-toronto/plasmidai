#!/bin/bash

source $HOME/envs/wandb-env/bin/activate
export REPO_ROOT=~/code/plasmid-ai
cd $REPO_ROOT
wandb sync logs/wandb/offline-*
