from Bio import SeqIO

path = "/home/linxin67/scratch/IGEM_data/plasmids.fasta"
L = 1000
# for r in SeqIO.parse(path, 'fasta'):
#     print(r)
records = [r for r in SeqIO.parse(path, "fasta") if len(r.seq) < L]
print("Filtered:", len(records))
print(records)
print(type(records))
print(type(records[0]))
# with open(path) as handle:
#     for values in SimpleFastaParser(handle):
#         print(values)
