from argparse import ArgumentParser
from orffinder import orffinder
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import glob
from tqdm import tqdm
join = os.path.join
from math import ceil
import subprocess

def json_to_fasta(input_json: str,
                  output_folder: str,
                  size_per_fasta_file: int=1000):
    """Extract DNA sequences from input .json file into fasta files 
    of size_per_fasta_file lines to optimize with running `amrfinderplus` later. 
    Default size=1000"""
    os.makedirs(join(root_dir, output_folder), exist_ok=True)
    json_dict = {}
    with open(join(root_dir, input_json), 'r') as f:
        json_dict = json.load(f)
    
    total = len(json_dict['data'])
    for i in tqdm(range(1, ceil(total//size_per_fasta_file))):
      with open(join(root_dir,  output_folder, f'{input_json}_part{str(i)}.fasta'), 'w') as f:
        for j, row in enumerate(json_dict['data']):
            if (i-1)*1000 < j < i*1000:
              f.write(f'>sp{j}\n{row[0]}\n')


def run_amrfinder(input_folder):
  temp_processed_folder_name = f'processed_{os.path.basename(input_folder)}'
  output_folder = join(os.path.dirname(input_folder, temp_processed_folder_name))
  os.makedirs(output_folder, exist_ok=True)
  for filename in tqdm(os.listdir(input_folder)):
      filepath = os.path.join(input_folder, filename)
      if os.path.isfile(filepath):
          output_seq_dir = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.fasta")
          output_metadata_dir = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.tsv")
          with open(output_metadata_dir, 'w') as f:
            command = f"!amrfinder -n ${filepath} \
                      --plus \
                      --threads 4 \
                      -O Escherichia \
                      --nucleotide_flank5_output ${output_seq_dir} \
                      --nucleotide_flank5_size 100 \
                      --coverage_min 0.2 \
                      > ${output_metadata_dir}"
            args = args.split(" ")
            subprocess.run(command)
              
  # merge results into one .tsv file
  with open(join(root_dir, "amrfinder_output.tsv"), "w") as outfile:
    included_header = False
    for in_path in glob.glob(join(root_dir, "amrfinder_output", "*.tsv")):
      header_line = True
      with open(in_path) as infile:
        for line in infile:
          if header_line:
            if not included_header:
              outfile.write(line)
              included_header = True
            header_line = False
          else:
            outfile.write(line)
  aa_seq_df = pd.read_csv(join(root_dir, "selected_amrfinder_output.tsv"), sep="\t")  
  return aa_seq_df


def visualize_metadata(df: pd.DataFrame):
  fig, ax = plt.subplots(figsize=(6, 4))
  aa_seq_df['Alignment length'].plot(kind='hist', density=True, bins=20)
  aa_seq_df['Alignment length'].plot(kind='kde')
  ax.set_xlabel('Alignment length')
  ax.set_ylabel('Frequency')
  ax.set_title('Alignment length distribution')
  plt.show()
  plt.save(join(root_dir, 'alignment_length_distribution.png'))

  fig, ax = plt.subplots(figsize=(6, 4))
  aa_seq_df['% Coverage of reference sequence'].plot(kind='hist', density=False, bins=15)
  ax.set_xlabel('% Coverage of reference sequence')
  ax.set_ylabel('Count')
  ax.set_title('% Coverage of reference sequence')
  plt.show()
  plt.save(join(root_dir, 'coverage_distribution.png'))

  fig, ax = plt.subplots(figsize=(6, 4))
  aa_seq_df['% Identity to reference sequence'].plot(kind='hist', density=False, bins=7)
  ax.set_xlabel('% Identity to reference sequence')
  ax.set_ylabel('Count')
  ax.set_title('Percent match distribution')
  plt.show()
  plt.save(join(root_dir, 'identity_distribution.png')) 



def filter_dna_to_aa(input_folder: str, output_dir: str, aa_seq_df: pd.DataFrame):
    input_folder = join(root_dir, 'amrfinder_output')
    uniq_id = []
    for i in range(len(aa_seq_df)):
      row = aa_seq_df.iloc[i]
      id = f"{row.iloc[1]}:{int(row.iloc[2] - 100) if int(row.iloc[2]) > 100 else row.iloc[2]}-{row.iloc[3]}"
      uniq_id.append(id)
    aa_seq_df.insert(0, "Unique ID", uniq_id, True)
    print("Number of selected sequences: ", len(uniq_id))
    
    final_aa_seqs = []
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w") as outfile:
      for filename in os.listdir(input_folder):
        if filename.endswith('.fasta'):
          input_dir = os.path.join(input_folder, filename)
          for dna_record in SeqIO.parse(input_dir, "fasta"):
            # use both forward and reverse complement sequences
            try:
              if dna_record.id in uniq_id:
                dna_seqs = [dna_record.seq, dna_record.seq.reverse_complement()]
              # generate all translation frames
                aa_seqs = (s[i:].translate(to_stop=True) for i in range(3) for s in dna_seqs)
                selected_aa = max(aa_seqs, key=len)
                if  len(selected_aa) > 300:
                  print("length of selected amino acid sequences: ", len(selected_aa))
                  aa_record = SeqRecord(selected_aa, id=dna_record.id, description=dna_record.description)
                  SeqIO.write(aa_record, outfile, "fasta")
                  final_aa_seqs.append(dna_record.id)
            except Exception as e:
              print(f'Error {e} at DNA seq id {dna_record.id} in file {filename}')
              continue
    final_seq_df = aa_seq_df[aa_seq_df['Unique ID'].isin(final_aa_seqs)]
    final_seq_df.to_csv(join(root_dir, 'final_finetune_aa_seq.tsv'), sep="\t", index=False)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
    "--input_json",
    type=str,
    default="finetune_2024-09-07-10-59_samples.json",
    help="filename of input .json file containing sample whole plasmid DNA sequences"
  )
  parser.add_argument(
    "--dna_fasta_dir",
    type=str,
    default="selected_dna_fasta_samples",
    help="directory for processed .fasta files"
  )  
  parser.add_argument(
    "--protein_output_name",
    default="selected_proteins.fasta",
    help="preferred name for selected protein output file"
  )

  args = parser.parse_args()
  json_to_fasta(args.input_json, args.fasta_dir)
  selected_df = run_amrfinder(args.fasta_dir)
  visualize_metadata(selected_df)
  filter_dna_to_aa(args.protein_output_name, aa_seq_df=selected_df)
  
