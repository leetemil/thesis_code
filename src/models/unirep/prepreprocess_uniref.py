from pathlib import Path

from Bio import SeqIO
source_file = Path("../../data/files/uniref50.fasta")
target_file = open(Path("../../data/files/uniref50_short.fasta"), "w")
seqs = SeqIO.parse(source_file, "fasta")

count = 0
for i, seq in enumerate(seqs):
    if i % 1000000 == 0:
        print(f"{i}")
    if len(seq) <= 2000:
        SeqIO.write(seq, target_file, "fasta")
        count += 1

print(f"Count: {count}")
target_file.close()
