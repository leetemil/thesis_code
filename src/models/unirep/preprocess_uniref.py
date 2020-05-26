import random
from pathlib import Path

from Bio import SeqIO

# num_proteins = 38_836_200
# num_proteins = 38_695_195
num_proteins = 39_069_211
train_ratio = 0.99
val_ratio = 1 - train_ratio

train_len = int(train_ratio * num_proteins)
val_len = num_proteins - train_len

val_indices = sorted(random.sample(range(num_proteins), val_len))
current_val_idx = 0

source_file = Path("../../data/files/uniref50_short.fasta")
train_file = open(source_file.with_name(source_file.stem + "_train" + source_file.suffix), "w")
val_file = open(source_file.with_name(source_file.stem + "_validation" + source_file.suffix), "w")

train_file.write(f"# {train_len}\n")
val_file.write(f"# {val_len}\n")

seqs = SeqIO.parse(source_file, "fasta")

for i, seq in enumerate(seqs):
    if i % 1000000 == 0:
        print(f"{i}")
    if i != val_indices[current_val_idx]:
        SeqIO.write(seq, train_file, "fasta")
    else:
        current_val_idx = min(current_val_idx + 1, len(val_indices) - 1)
        SeqIO.write(seq, val_file, "fasta")

train_file.close()
val_file.close()
