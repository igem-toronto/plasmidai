import yaml
import os
import torch
import wandb
import random
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def open_config(file_name, folder_path='configs'):
    '''Open the configuration file and return a dictionary'''
    with open(os.path.join(folder_path, file_name), "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("[INFO] Read successful, here are the characteristics of your model: ")
    print(data)
    return data

def set_random_seed(seed):
    """Sets random seed for training reproducibility
    
    Args:
        seed (int)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_directory_if_not_exists(path):
    '''Creates a directory if such a directory does not exist'''
    if not os.path.exists(path):
        os.makedirs(path)

def count_param_numbers(model):
    '''Returns the number of parameters in a given model
    
    Args:
        model (nn.Module)
    '''
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params

def save_checkpoint(epoch, model, lr, optimizer, show_val_loss, wandb_id, checkpoint_path):
    '''Save the checkpoint at the given path with the following information
    '''
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'show_val_loss': show_val_loss,
        'wandb_id': wandb_id
        }, checkpoint_path)

if __name__ == '__main__':
    open_config('Resized_testing_heatmap_beluga.yaml')
