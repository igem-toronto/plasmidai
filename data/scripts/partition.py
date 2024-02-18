import argparse

import pandas as pd
from Bio import SeqIO

SEED = 16311
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}


def will_finetune(r):
    desc = r.description.lower()
    return ("escherichia coli" in desc) and (len(r.seq) < 10000)


def rounded_split(n):
    split = {k: int(n * p) for k, p in SPLIT_RATIO.items()}
    leftover = n - sum(split.values())
    split["train"] += leftover
    return split


def partition_plasmids(path, clusters, out):
    columns = ["centroid", "id"]
    df = pd.read_csv(clusters, sep=r"\s+", names=columns)

    records = {r.id: r for r in SeqIO.parse(path, "fasta")}
    singletons = set(records) - set(df["id"].unique())
    singletons = pd.DataFrame([[i, i] for i in singletons], columns=columns)
    df = pd.concat([df, singletons])

    df["n"] = 1
    df["nf"] = df["id"].apply(lambda k: will_finetune(records[k]))
    df["id"] = df["id"].apply(lambda k: k + ",")  # makes it easy to recover IDs

    df = df.groupby("centroid").sum()
    df = df.sort_values(by="centroid")

    assgn = dict()
    counts = {k: {"nf": 0, "n": 0} for k in SPLIT_RATIO}
    totals = df.sum()

    for col in ["nf", "n"]:
        subdf = df[df[col] > 0]
        subdf = subdf.sample(frac=1, random_state=SEED)

        cutoffs = rounded_split(totals[col])
        for _, cluster in subdf.iterrows():
            dnf = cluster["nf"]
            for split in ["test", "val", "train"]:
                if (counts[split][col] + dnf) <= cutoffs[split]:
                    break
            for sid in cluster["id"].split(",")[:-1]:
                assert sid in records
                assgn[sid] = {"id": sid, "split": split}
                counts[split]["n"] += 1
            counts[split]["nf"] += dnf

        df = df[df[col] == 0]  # what remains

    assert len(assgn) == len(records)
    assgn = pd.DataFrame(list(assgn.values()))
    assgn["finetune"] = assgn["id"].apply(lambda k: will_finetune(records[k]))
    assgn = assgn.sort_values(by="id").set_index("id")
    assgn.to_csv(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--clusters", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    partition_plasmids(**vars(args))
