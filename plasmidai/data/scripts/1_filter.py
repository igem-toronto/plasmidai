import argparse
from typing import List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def filter_below_length(path: str, L: int) -> None:
    """
    Filter FASTA sequences below a specified length and write them to a new file.

    This function reads sequences from a FASTA file, filters out sequences
    shorter than the specified length, and writes the filtered sequences
    to a new FASTA file with '.short.fasta' appended to the original filename.

    Args:
        path (str): The path to the input FASTA file.
        L (int): The length threshold for filtering sequences.

    Raises:
        AssertionError: If the input file does not have a '.fasta' extension.

    Side effects:
        - Prints the number of filtered sequences.
        - Writes filtered sequences to a new FASTA file.
    """
    assert path.endswith(".fasta"), "Input file must have a .fasta extension"

    records: List[SeqRecord] = [r for r in SeqIO.parse(path, "fasta") if len(r.seq) < L]
    print(f"Filtered: {len(records)}")

    output_path: str = path.replace(".fasta", ".short.fasta")
    SeqIO.write(records, output_path, "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter FASTA sequences below a specified length."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the input FASTA file"
    )
    parser.add_argument(
        "--L",
        type=int,
        default=15000,
        help="Length threshold for filtering (default: 15000)",
    )
    args = parser.parse_args()

    filter_below_length(**vars(args))
