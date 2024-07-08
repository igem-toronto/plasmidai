import collections

import jsonargparse
import numpy as np
import pandas as pd
from Bio import SeqIO

SEED = 16311


def will_finetune(r, path):
    if "plasmids" in path:
        desc = r.description.lower()
        return ("escherichia coli" in desc) and (len(r.seq) < 10000)
    elif "replicons" in path:
        return False
    else:
        raise ValueError()


def split_indices(n):  # do an 8:1:1 split
    indices = collections.defaultdict(list)

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


def partition(path: str, clusters: str, out: str):
    columns = ["centroid", "id"]
    df = pd.read_csv(clusters, sep=r"\s+", names=columns)

    records = {r.id: r for r in SeqIO.parse(path, "fasta")}
    singletons = set(records) - set(df["id"].unique())
    singletons = pd.DataFrame([[i, i] for i in singletons], columns=columns)
    df = pd.concat([df, singletons])

    df["n"] = 1
    df["id"] = df["id"].apply(lambda k: k + ",")  # makes it easy to recover IDs

    df = df.groupby("centroid").sum()
    df = df.sort_values(by="n", ascending=False)
    df = df.reset_index()

    assgn = dict()
    for split, indices in split_indices(len(df.index)).items():
        for idx in indices:
            cluster = df.iloc[idx]
            for sid in cluster["id"].split(",")[:-1]:
                assert (sid in records) and (sid not in assgn)
                assgn[sid] = {"id": sid, "cluster": idx, "split": split}

    assert len(assgn) == len(records)
    assgn = pd.DataFrame(list(assgn.values()))
    assgn["finetune"] = assgn["id"].apply(lambda k: will_finetune(records[k], path))
    assgn = assgn.sort_values(by="id").set_index("id")
    assgn.to_csv(out)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(partition)
    args = parser.parse_args()

    partition(**vars(args))
