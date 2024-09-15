'''Downloading the weights from Evo on hugginface'''
import torch
# Load model directly
from transformers import AutoModelForCausalLM
import os

def download_evo_weights(destination_dir):
    """
    Download the Evo weights from Hugging Face and save it to a specified directory
    without loading it into RAM.
    https://huggingface.co/togethercomputer/evo-1-8k-base/tree/main

    Args:
        destination_dir (str): The directory where the dataset should be saved.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Download and load the model
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/evo-1-8k-base", trust_remote_code=True)

    model_save_path = os.path.join(destination_dir, 'evo-1-8k-base.pth')

    # Save the model locally
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")

if __name__ == '__main__':
    # model_save_path = "/home/xinleilin/Projects/IGEM/plasmid-ai/src/experimental/evo-fine-tune/checkpoints/evo-1-8k-base"
    model_save_path = '/home/linxin67/projects/def-mikeuoft/linxin67/Projects/plasmid-ai/src/experimental/evo-fine-tune/checkpoints/evo-1-8k-base'
    download_evo_weights(model_save_path)
