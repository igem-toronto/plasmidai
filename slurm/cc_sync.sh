#!/bin/bash

module load StdEnv/2023 python/3.10 scipy-stack

# Add your project directory path here
export PROJECT=# <YOUR_PROJECT_DIRECTORY>

# Add the path to your virtual environment
source # <PATH_TO_YOUR_VIRTUAL_ENV>/bin/activate

# Add the path to your repository root
export REPO_ROOT=# <YOUR_REPO_ROOT_PATH>

cd $REPO_ROOT

# Watch and sync WandB logs every 300 seconds (5 minutes)
watch -n 300 wandb sync $REPO_ROOT/logs/wandb/offline-*