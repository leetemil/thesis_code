from pathlib import Path
import argparse

from datetime import datetime
import pickle
import math

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from Bio import SeqIO

from models import VAE
from data import get_datasets, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq

parser = argparse.ArgumentParser(description = "Make some figures for the provided model", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = "@")
parser.add_argument("--model_path", type = Path, default = Path("results/BLAT_ECOLX_Ranganathan2015/model.torch"), help = "Model path.")
args = parser.parse_args()

PICKLE_FILE = Path('data/files/mutation_data.pickle')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
load_dict = torch.load(args.model_path, map_location = device)
name = load_dict["name"]
state_dict = load_dict["state_dict"]
args_dict = load_dict["args_dict"]
model_type = VAE
model = model_type(**args_dict).to(device)
model.load_state_dict(state_dict)

# Load data
sheet = "BLAT_ECOLX_Ranganathan2015"
data_path = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m")

with open(PICKLE_FILE, 'rb') as f:
    proteins = pickle.load(f)
    p = proteins[sheet].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

sequences = SeqIO.parse(data_path, "fasta")
wt_seq = next(sequences)
other_seq = next(sequences)

chosen_seq = wt_seq

wt = seq2idx(chosen_seq, device)
encoded = model.encode(wt.unsqueeze(0)).mean
decoded = model.sample(encoded)

in_seq = str(chosen_seq.seq).upper()
out_seq = idx2seq(decoded.squeeze().cpu().numpy()).replace('<mask>', '.')

accuracy = sum(a1 == a2 for (a1, a2) in zip(in_seq, out_seq)) / len(in_seq)

print(f'Encoded representation (mean of encoder distribution):\n{encoded.squeeze()}\n')
print(f'{in_seq}')
print(f'{out_seq}')
print(accuracy)

