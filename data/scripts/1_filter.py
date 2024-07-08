import argparse

from Bio import SeqIO


def filter_below_length(path, L):
    assert path.endswith(".fasta")

    records = [r for r in SeqIO.parse(path, "fasta") if len(r.seq) < L]
    print("Filtered:", len(records))

    SeqIO.write(records, path.replace(".fasta", ".short.fasta"), "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--L", type=int, default=15000)
    args = parser.parse_args()

    filter_below_length(**vars(args))
