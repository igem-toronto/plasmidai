# Adapted from "https://github.com/ncbi/amr"

1. Install bioconda according to instructions: https://bioconda.github.io/
2. Run the following lines in terminal to set up conda virtual environment and install requirements

`conda config --add channels bioconda`\
`conda config --add channels conda-forge`\
`conda config --set channel_priority strict`\

`conda create -y -c conda-forge -c bioconda -n amrfinder --strict-channel-priority ncbi-amrfinderplus`\
`conda init`\
`source activate amrfinder`\
`conda install blast ncbi-amrfinderplus blast hmmer biopython`\
`pip install orffinder`\
`amrfinder -u`\

3. Export global environment variables
`export root_dir=<path_to_root_dir>`\

4. To extract protein sequences from .json file containing DNA sequences, run:
`python amrfinder_runner.py --input_json <input_json_file> --dna_fasta_dir <dir_to_selected_dna>`

An output file named `selected_proteins.fasta` will be generated, containing the selected proteins found by AMRFinderPlus. 
This name can be overwritten with the `--protein_output_name` flag.

5. To run ESMFold 3D structure predictions with the generated protein sequences, run:
`python esmfold_runner.py selected_proteins.fasta`

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



