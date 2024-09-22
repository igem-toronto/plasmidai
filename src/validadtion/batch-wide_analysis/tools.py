import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, Align
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def visualize(data: pd.DataFrame, folder: str, sufix: str = "") -> None:
    """
    Visualize various aspects of the data and save the plots.

    Args:
        data (pd.DataFrame): The input data to visualize.
        folder (str): The folder path to save the plots.
        sufix (str, optional): A suffix to add to the filenames. Defaults to "".
    """
    # Display summary statistics
    print(data.describe())

    # Histogram for 'size'
    plt.figure(figsize=(10, 6))
    sns.histplot(data["size"], kde=True)
    plt.title("Distribution of Size")
    plt.xlabel("Size")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(folder, f"size_distribution2_{sufix}.png"))
    plt.show()

    # Histogram for 'gc'
    plt.figure(figsize=(10, 6))
    sns.histplot(data["gc"], kde=True)
    plt.title("Distribution of GC Content")
    plt.xlabel("GC Content")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(folder, f"gc_distribution_{sufix}.png"))
    plt.show()

    # Scatter plot of 'size' vs 'gc'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="size", y="gc", data=data)
    plt.title("Size vs GC Content")
    plt.xlabel("Size")
    plt.ylabel("GC Content")
    plt.savefig(os.path.join(folder, f"size_vs_gc_{sufix}.png"))
    plt.show()

    # Frequency plot for rep_type, but don't count the values with value '-'
    plt.figure(figsize=(10, 6))
    sns.countplot(x="rep_type(s)", data=data[data["rep_type(s)"] != "-"])
    plt.title("Repeat Type")
    plt.xlabel("Repeat Type")
    plt.ylabel("Count")
    plt.savefig(os.path.join(folder, f"repeat_type{sufix}.png"))
    plt.show()

    # Box plot for 'size' vs 'rep_type'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="rep_type(s)", y="size", data=data)
    plt.title("Size vs Repeat Type")
    plt.xlabel("Repeat Type")
    plt.ylabel("Size")
    plt.savefig(os.path.join(folder, f"size_vs_repeat_type_{sufix}.png"))
    plt.show()

    # Frequency plot of orit_type(s)
    plt.figure(figsize=(10, 6))
    sns.countplot(x="orit_type(s)", data=data[data["orit_type(s)"] != "-"])
    plt.title("Origin Type")
    plt.xlabel("Origin Type")
    plt.ylabel("Count")
    plt.savefig(os.path.join(folder, f"origin_type{sufix}.png"))
    plt.show()


def align_sequences(
    seq1: SeqRecord, seq2: SeqRecord, match_score: int = 1, mismatch_score: int = -2
) -> tuple[float, float]:
    """
    Performs both global and local alignments on two sequences.

    Args:
        seq1 (SeqRecord): The first sequence to align.
        seq2 (SeqRecord): The second sequence to align.
        match_score (int, optional): Score for matching bases. Defaults to 1.
        mismatch_score (int, optional): Score for mismatching bases. Defaults to -2.

    Returns:
        tuple[float, float]: A tuple of (global_score, local_score).

    Notes:
        - A pairwise alignment is global if it is known that the sequences are homologous in their full length.
        - A local alignment is needed if it is known that one sequence is shorter than the other
          and that it cannot be related to the other in its full length.
        - The alignment length is defined as the number of columns in the alignment as printed.
          This is equal to the sum of the number of matches, number of mismatches, and the total length
          of gaps in the target and query.
    """
    aligner = Align.PairwiseAligner()
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score

    # Global alignment
    aligner.mode = "global"
    global_align_results = aligner.align(seq1, seq2)
    global_score = global_align_results.score

    # Local alignment
    aligner.mode = "local"
    local_alignment_results = aligner.align(seq1, seq2)
    local_score = local_alignment_results.score

    return global_score, local_score


def allign_mobsuite_results(
    mobsuite_output_dir: str,
    output_dir: str,
    database: str = "curated_databases/doriC_ori.fasta",
) -> pd.DataFrame:
    """
    Aligns all contigs in the mobsuite output directory to the database.

    Args:
        mobsuite_output_dir (str): The directory containing the mobsuite output.
        output_dir (str): The directory to save the alignment scores.
        database (str, optional): The database to align the contigs to. Defaults to "curated_databases/doriC_ori.fasta".

    Returns:
        pd.DataFrame: A DataFrame containing the alignment results.
    """
    # Load the database
    db = list(SeqIO.parse(database, "fasta"))
    print(f"Loaded {len(db)} sequences from the database")
    alignment_results_df = pd.DataFrame(
        columns=["Query", "Database", "Global_Score", "Local_Score", "Cluster"]
    )

    for folder in os.listdir(mobsuite_output_dir):
        print(f"Processing {folder}")
        for file in os.listdir(os.path.join(mobsuite_output_dir, folder)):
            if "plasmid" in file:
                file_path = os.path.join(mobsuite_output_dir, folder, file)
                print(f"Processing cluster {file}")
                contigs = list(SeqIO.parse(file_path, "fasta"))
                for contig in tqdm(contigs, desc=f"Processing {file}"):
                    for db_seq in db:
                        global_score, local_score = align_sequences(contig, db_seq)
                        alignment_results_df = alignment_results_df._append(
                            {
                                "Query": contig.id,
                                "Database": db_seq.id,
                                "Global_Score": global_score,
                                "Local_Score": local_score,
                                "Cluster": file,
                            },
                            ignore_index=True,
                        )
                print(alignment_results_df)

    alignment_results_df.to_csv(
        os.path.join(output_dir, "alignment_scores.csv"), index=False
    )
    return alignment_results_df
