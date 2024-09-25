# Adapted from https://github.com/facebookresearch/esm
from esmfold_utils import *
from Bio import SeqIO
from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
import gc
import os

# install ESMFold, OpenFold
version = "1" # ["0", "1"]
model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
import os, time
if not os.path.isfile(model_name):
  # download esmfold params
  os.system("apt-get install aria2 -qq")
  os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &")

  if not os.path.isfile("finished_install"):
    # install libs
    print("installing libs...")
    os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif")
    os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")

    print("installing openfold...")
    # install openfold
    os.system(f"pip install -q git+https://github.com/sokrypton/openfold.git")

    print("installing esmfold...")
    # install esmfold
    os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")
    os.system("touch finished_install")

  # wait for Params to finish downloading...
  while not os.path.isfile(model_name):
    time.sleep(5)
  if os.path.isfile(f"{model_name}.aria2"):
    print("downloading params...")
  while os.path.isfile(f"{model_name}.aria2"):
    time.sleep(5)
    
root_dir = '/content/drive/MyDrive/igem-2024'
join = os.path.join

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]

  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  return o

def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
alphabet_list = list(ascii_uppercase+ascii_lowercase)


def run_esmfold(input_dir: str):
  output_folder = os.path.join(root_dir, 'esmfold_predictions')
  os.makedirs(output_folder, exist_ok=True)
  record_dict = SeqIO.to_dict(SeqIO.parse(input_dir, "fasta"))
  for id, seq_obj in record_dict.items():
    print(f"Predicting sequence {id}")
    print(seq_obj.description)
    jobname = f"test_{id}"

    sequence = str(seq_obj.seq)
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    copies = 1 #param {type:"integer"}
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 6 # ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
    chain_linker = 25

    ID = jobname+"_"+get_hash(sequence)[:3]
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length",length)

    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"

    if "model" not in dir() or model_name != model_name_:
      if "model" in dir():
        # delete old model from memory
        del model
        gc.collect()
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

      model = torch.load(model_name)
      model.eval().cuda().requires_grad_(False)
      model_name_ = model_name

    # optimized for Tesla T4
    if length > 700:
      model.set_chunk_size(64)
    else:
      model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(sequence,
                        num_recycles=num_recycles,
                        chain_linker="X"*chain_linker,
                        residue_index_offset=512)

    pdb_str = model.output_to_pdb(output)[0]
    output = tree_map(lambda x: x.cpu().numpy(), output)
    ptm = output["ptm"][0]
    plddt = output["plddt"][0,...,1].mean()
    O = parse_output(output)
    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
    os.makedirs(join(output_folder, ID), exist_ok=True)
    prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
    with open(f"{output_folder}/{prefix}_pae.txt", 'w') as outfile:
      np.savetxt(outfile, O["pae"], "%.3f")
    with open(f"{output_folder}/{prefix}.pdb","w") as out:
      out.write(pdb_str)

    show_pdb(pdb_str, color=color,
         show_sidechains=show_sidechains,
         show_mainchains=show_mainchains,
         Ls=lengths).show()

    plot_confidence(O, Ls=lengths, dpi=dpi)
    plt.savefig(f'{output_folder}/{prefix}.png',bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    selected_amr_aa_seq = join(root_dir, 'selected_finetune_aa_seq.fasta')
    run_esmfold(selected_amr_aa_seq)

    
