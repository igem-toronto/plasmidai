import argparse
import os

import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback


from utils.tools import open_config, set_random_seed, create_directory_if_not_exists, count_param_numbers, save_checkpoint
from model.LitEvo import LitEvo
from data_format.DataFormat import PlasmidDataset


def parse_args(file_path='configs'):
    '''
    Parse arguments passed with training file, and returns the corresponding config file in the format of a dictionary
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='evo-fine-tune/soroush_basemodel.yaml',
                        help='Name of the configuration file')
    args = parser.parse_args()
    config_file = args.config

    # import configurations:
    return open_config(config_file, file_path)

# class GradientLoggingCallback(Callback):
#     def __init__(self, log_interval):
#         super().__init__()
#         self.log_interval = log_interval

#     def on_after_backward(self, trainer, pl_module):
#         if trainer.current_epoch % self.log_interval == 0:
#             for name, param in pl_module.named_parameters():
#                 if param.grad is not None:
#                     wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})

def train(config):
    '''Train the model

    Args:
        config (dict): configuration dictionary
    '''
    train_set = PlasmidDataset(config, train_set=True,
                               real_job=config['real_job'], tokenizer=None)
    val_set = PlasmidDataset(config, train_set=False,
                             real_job=config['real_job'], tokenizer=None)

    num_cpu_cores = os.cpu_count()
    active_gpus = torch.cuda.device_count()
    # num_workers = config['num_cpus'] * (num_cpu_cores) - 1
    num_workers = 1 # for cedar
    common_loader_params = {
        # 'batch_size': config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False
    }
    print(
        f'[INFO] num_workers: {num_workers} && num_cpus_cores: {num_cpu_cores} cores')

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True, **common_loader_params)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False, **common_loader_params)

    print(f'[INFO] Number of batches in train_set: {len(train_loader)}')
    print(f'[INFO] Number of batches in val_set: {len(val_loader)}')

    model = LitEvo(config)

    checkpoint_directory = os.path.join(
        config['checkpoint_directory'], config['checkpoint_name'])
    create_directory_if_not_exists(checkpoint_directory)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_directory,
        filename=config['checkpoint_name'] + "-{epoch:02d}-{val_loss:.2f}",
    )

    wandb_logger = WandbLogger(
        name = config['checkpoint_name'],
        offline = True,
        project = config['model_name'],
    )
    wandb_logger.watch(model, log_freq=config['wandb_grad_log_interval']) # log gradients

    # gradient_logging_callback = GradientLoggingCallback(log_interval=config['wandb_grad_log_interval'])

    trainer = L.Trainer(max_epochs=config['epoch_number'], accelerator='auto',
                        strategy='ddp', gradient_clip_val=config['max_grad_norm'], logger=wandb_logger)#, callbacks=[checkpoint_callback, gradient_logging_callback])
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


def main():
    config = parse_args()
    set_random_seed(config['seed'])
    train(config)


if __name__ == '__main__':
    main()
