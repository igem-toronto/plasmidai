import os
import json
import pandas as pd


def extract_dir_name(dir: str):
    i = 0
    while i < len(dir):
        i += 1
        if dir[-i] == "/" or dir[-i] == "\\":
            break
    return dir[:-i]
def whatever_to_fasta(filepath: str):
    print(f"Converting {filepath} to FASTA")
    if not os.path.isdir(filepath):
        # if the file is not a directory, then it is a file
        output_dir = extract_dir_name(filepath)
        output_name = "/" + filepath.split("/")[-1].split(".")[0] + ".fasta" if "/" in filepath else "\\" + filepath.split("\\")[-1].split(".")[0] + ".fasta"
        with open(filepath, "r") as file:
            print(f"Reading {filepath} \t writing {output_dir + output_name}")
            with open(output_dir + output_name, "w") as fasta:
                if filepath.endswith(".csv"):
                    # "C G C G G C C G G T T A C G G G G G A C A...","1174",...
                    file_df = pd.read_csv(file)
                    for num, seq in enumerate(file_df["sequence"]):
                        sequence = seq.strip()
                        fasta.write(f">Sequence_{num}\n")
                        fasta.write(f"{sequence}\n")
                elif filepath.endswith(".json"):
                    # {"columns": ["sequence"], "data": [["GACGATCCGCCTTTATGAAAGTCTTGCTCAATTC...], ...]}
                    data = json.load(file)
                    for num, seq in enumerate(data["data"]):
                        sequence = seq[0].strip()
                        fasta.write(f">Sequence_{num}\n")
                        fasta.write(f"{sequence}\n")
        print(f"FASTA file {output_dir} created successfully")
    else:
        # if the file is a directory, then it is a directory
        for element in os.listdir(filepath):
            if element.endswith(".csv") or element.endswith(".json"):
                whatever_to_fasta(os.path.join(filepath, element))


if __name__ == "__main__":
    sequences_dir = os.curdir
    for i in os.listdir(sequences_dir):
        if not i.endswith(".py") and not i.endswith(".fasta") and not i.endswith(".sh"):
            whatever_to_fasta(os.path.join(sequences_dir, i))