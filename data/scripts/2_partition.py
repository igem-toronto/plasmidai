import collections
from typing import Dict, List, DefaultDict

import jsonargparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

SEED: int = 23

def will_finetune(r: SeqRecord, path: str) -> bool:
    """
    Determine if a sequence record should be used for fine-tuning based on its source and characteristics.

    Args:
        r (SeqRecord): The sequence record to evaluate.
        path (str): The path of the input file, used to determine the sequence type.

    Returns:
        bool: True if the sequence should be used for fine-tuning, False otherwise.

    Raises:
        ValueError: If the path doesn't contain 'plasmids' or 'replicons'.
    """
    if "plasmids" in path:
        desc = r.description.lower()
        return ("escherichia coli" in desc) and (len(r.seq) < 10000)
    elif "replicons" in path:
        return False
    else:
        raise ValueError("Path must contain 'plasmids' or 'replicons'")

def split_indices(n: int) -> DefaultDict[str, List[int]]:
    """
    Split indices into train, validation, and test sets with an 8:1:1 ratio.

    Args:
        n (int): Total number of indices to split.

    Returns:
        DefaultDict[str, List[int]]: A dictionary with keys 'train', 'val', and 'test',
                                     containing the respective indices.
    """
    indices: DefaultDict[str, List[int]] = collections.defaultdict(list)

    rng = np.random.default_rng(SEED)
    stride = 10
    for i in range(0, n, stride):
        batch = list(range(i, min(n, i + stride)))
        if len(batch) < stride:  # last batch
            indices["train"] += batch
        else:
            if i > 0:  # don't shuffle first batch
                rng.shuffle(batch)
            indices["train"] += batch[:-2]
            indices["val"].append(batch[-2])
            indices["test"].append(batch[-1])

    return indices

def partition(path: str, clusters: str, out: str) -> None:
    """
    Partition sequences into train, validation, and test sets based on cluster information.

    Args:
        path (str): Path to the input FASTA file.
        clusters (str): Path to the cluster information file.
        out (str): Path to save the output CSV file.

    Side effects:
        - Writes a CSV file with sequence partitions and fine-tuning information.
    """
    columns: List[str] = ["centroid", "id"]
    df: pd.DataFrame = pd.read_csv(clusters, sep=r"\s+", names=columns)

    records: Dict[str, SeqRecord] = {r.id: r for r in SeqIO.parse(path, "fasta")}
    singletons: set = set(records) - set(df["id"].unique())
    singletons_df: pd.DataFrame = pd.DataFrame([[i, i] for i in singletons], columns=columns)
    df = pd.concat([df, singletons_df])

    df["n"] = 1
    df["id"] = df["id"].apply(lambda k: k + ",")  # makes it easy to recover IDs

    df = df.groupby("centroid").sum()
    df = df.sort_values(by="n", ascending=False)
    df = df.reset_index()

    assgn: Dict[str, Dict[str, Union[str, int]]] = {}
    for split, indices in split_indices(len(df.index)).items():
        for idx in indices:
            cluster = df.iloc[idx]
            for sid in cluster["id"].split(",")[:-1]:
                assert (sid in records) and (sid not in assgn)
                assgn[sid] = {"id": sid, "cluster": idx, "split": split}

    assert len(assgn) == len(records)
    assgn_df: pd.DataFrame = pd.DataFrame(list(assgn.values()))
    assgn_df["finetune"] = assgn_df["id"].apply(lambda k: will_finetune(records[k], path))
    assgn_df = assgn_df.sort_values(by="id").set_index("id")
    assgn_df.to_csv(out)

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(partition)
    args = parser.parse_args()

    partition(**vars(args))