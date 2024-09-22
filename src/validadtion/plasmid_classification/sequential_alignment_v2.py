from typing import List, Dict, Any
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
import csv
import concurrent.futures
from tqdm import tqdm


def align_sequences(
    seq1: str,
    seq2: str,
    match: float = 2,
    mismatch: float = -1,
    gap_open: float = -2,
    gap_extend: float = -0.5,
) -> float:
    """
    Align two sequences and return the best normalized alignment score.

    Args:
        seq1 (str): First sequence to align.
        seq2 (str): Second sequence to align.
        match (float): Score for matching bases. Defaults to 2.
        mismatch (float): Score for mismatching bases. Defaults to -1.
        gap_open (float): Penalty for opening a gap. Defaults to -2.
        gap_extend (float): Penalty for extending a gap. Defaults to -0.5.

    Returns:
        float: The best normalized alignment score.
    """
    alignments = pairwise2.align.localxx(seq1, seq2)
    best_alignment = alignments[0]
    alignment_length = best_alignment[4]  # Length of the alignment
    score = best_alignment[2]
    normalized_score = score / alignment_length  # Normalize by alignment length
    return normalized_score


def calculate_gc_content(seq: str) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        seq (str): DNA sequence.

    Returns:
        float: GC content as a percentage.
    """
    g = seq.count("G")
    c = seq.count("C")
    return 100 * (g + c) / len(seq)


def process_plasmid(
    plasmid: SeqRecord, elements: List[SeqRecord], threshold: float
) -> Dict[str, Any]:
    """
    Process a plasmid by aligning it with elements and calculating scores.

    Args:
        plasmid (SeqRecord): The plasmid to process.
        elements (List[SeqRecord]): List of elements to align with the plasmid.
        threshold (float): Threshold for considering an alignment significant.

    Returns:
        Dict[str, Any]: Dictionary containing alignment scores and other information.
    """
    plasmid_scores = {"sequence_name": plasmid.id}
    total_score = 0

    # Calculate GC content
    gc_content = calculate_gc_content(str(plasmid.seq))
    plasmid_scores["gc_content"] = gc_content

    for element in elements:
        normalized_score = align_sequences(str(plasmid.seq), str(element.seq))
        plasmid_scores[element.id] = normalized_score

        if normalized_score >= threshold:
            total_score += normalized_score

    plasmid_scores["total_score"] = total_score
    return plasmid_scores


def sequential_alignment(
    unknown_plasmid_file: str, elements_file: str, threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Perform sequential alignment of unknown plasmids against a set of elements.

    Args:
        unknown_plasmid_file (str): Path to the file containing unknown plasmids.
        elements_file (str): Path to the file containing elements for alignment.
        threshold (float): Threshold for considering an alignment significant. Defaults to 0.8.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing alignment results for each plasmid.
    """
    unknown_plasmids = list(SeqIO.parse(unknown_plasmid_file, "fasta"))
    elements = list(SeqIO.parse(elements_file, "fasta"))

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_plasmid, plasmid, elements, threshold): plasmid
            for plasmid in unknown_plasmids
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing plasmids",
        ):
            results.append(future.result())

    # Sort the results based on total_score, priority to P2 and GC content
    results = sorted(
        results,
        key=lambda x: (x["total_score"], x.get("P2", 0), x["gc_content"]),
        reverse=True,
    )

    return results


def save_to_csv(
    results: List[Dict[str, Any]], output_file: str = "plasmid_alignment_results.csv"
) -> None:
    """
    Save alignment results to a CSV file.

    Args:
        results (List[Dict[str, Any]]): List of dictionaries containing alignment results.
        output_file (str): Name of the output CSV file. Defaults to "plasmid_alignment_results.csv".
    """
    if results:
        fieldnames = (
            ["sequence_name", "gc_content"]
            + [
                key
                for key in results[0]
                if key not in ("sequence_name", "gc_content", "total_score")
            ]
            + ["total_score"]
        )

        with open(output_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    unknown_plasmid_file = "new_seqs_2024-07-26.fasta"
    elements_file = "elements.fasta"

    results = sequential_alignment(unknown_plasmid_file, elements_file, threshold=0.8)
    output_file_name = f"{unknown_plasmid_file.rsplit('.', 1)[0]}_local_alignment_results_normalized.csv"
    save_to_csv(results, output_file=output_file_name)

    print(f"Alignment results saved to '{output_file_name}'.")
