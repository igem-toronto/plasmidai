# Optimized Robust Protein Validation Pipeline

## Steps to Reproduce Results

1. **Install Bioconda**  
   Follow the [bioconda instructions](https://bioconda.github.io/) to set up your environment.

2. **Set up Conda Environment**
   Run the following commands in your terminal to create a virtual environment and install the required dependencies:
   ```bash
   conda config --add channels bioconda
   conda config --add channels conda-forge
   conda config --set channel_priority strict

   conda create -y -c conda-forge -c bioconda -n amrfinder --strict-channel-priority ncbi-amrfinderplus
   conda init
   source activate amrfinder
   conda install blast ncbi-amrfinderplus hmmer biopython
   pip install orffinder
   amrfinder -u

3. Export global environment variables: \
`export root_dir=<path_to_root_dir>`

4. To extract protein sequences from .json file containing DNA sequences, run: \
`python amrfinder_runner.py --input_json <input_json_file> --dna_fasta_dir <dir_to_selected_dna>`

An output file named `selected_proteins.fasta` will be generated, containing the selected proteins found by AMRFinderPlus. 
This name can be overwritten with the `--protein_output_name` flag.

5. To run ESMFold 3D structure predictions with the generated protein sequences, run:\
`python esmfold_runner.py selected_proteins.fasta`

## Acknowledgements
The pipeline was adapted from [NCBI AMRFinderPlus](https://github.com/ncbi/amr/) and [ESMFold](https://github.com/facebookresearch/esm)



