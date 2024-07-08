# Setup

Download the full-length plasmid dataset:

```bash
gdown "1iIsat00ST5vK-06BUstuTbJkfWKpV2lE" 
gdown "1HzHb-SSpA3IFOnwsloGUZzXX4bWr4fUB"
```


Unzip and rename it:

```bash 
gzip -d 240212_plasmid_seq_54646.fasta.gz
mv 240212_plasmid_seq_54646.fasta plasmids.fasta
mv 240702_nr_replicon_54646.fasta replicons.fasta
```

Everything below is **optional** since the `splits.csv` files are uploaded.
 
## Plasmids

Some plasmids are too long to cluster in reasonable time, so we will only try to cluster those <15 Kbp. This is
reasonable since we are interested in <10 Kbp *E. coli* plasmids. First, filter the plasmids by length:

```bash
python scripts/1_filter.py --path=plasmids.fasta --L=15000
```

Cluster the filtered plasmids with MMseqs2:

```bash
mmseqs easy-cluster plasmids.short.fasta plasmids tmp \
  --min-seq-id 0.8 --wrapped-scoring 1 --cov-mode 3 --cluster-reassign
```

Partition plasmids into training, validation, and test (80:10:10) sets:

```bash 
python scripts/2_partition.py --path=plasmids.fasta --clusters=plasmids_cluster.tsv --out=plasmids.splits.csv
```

## Replicons

Cluster the replicons with MMseqs2:

```bash
mmseqs easy-cluster replicons.fasta replicons tmp \
  --min-seq-id 0.9 --cov-mode 0 -c 0.7 --cluster-mode 0 --cluster-reassign
```

Partition plasmids into training, validation, and test (80:10:10) sets:

```bash 
python scripts/2_partition.py --path=replicons.fasta --clusters=replicons_cluster.tsv --out=replicons.splits.csv
```