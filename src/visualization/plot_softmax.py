from pathlib import Path
import random
import pickle

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from models import VAE
from data import ProteinDataset, NUM_TOKENS, IUPAC_AMINO_IDX_PAIRS

ALIGNPATH = Path('data/alignments')
BLAT_ECOL = 'BLAT_ECOLX_Palzkill2012'
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_1_b0.5.a2m')
BLAT_SEQ_FILE = ALIGNPATH / Path('BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m')
PICKLE_FILE = Path('data/mutation_data.pickle')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

protein_dataset = ProteinDataset(BLAT_SEQ_FILE, device)

model = VAE([7890, 1500, 1500, 30, 100, 2000, 7890], NUM_TOKENS).to(device)
model.load_state_dict(torch.load("model.torch", map_location=device)["state_dict"])

# Softmax for the first 4 proteins
fig, axes = plt.subplots(4)
for i, ax in enumerate(axes):
    if i == 0:
        idx = 0
    else:
        idx = random.randint(0, len(protein_dataset) - 1)

    sample, _, seq = protein_dataset[idx]
    mu, _ = model.encode(sample.unsqueeze(0))
    # z = model.reparameterize(mu, logvar)
    ds = model.decode(mu).squeeze(0).exp().cpu().detach().numpy()

    ax.imshow(ds.T, cmap=plt.get_cmap("Blues"))

    acids, _ = zip(*IUPAC_AMINO_IDX_PAIRS)
    ax.set_yticks(np.arange(len(IUPAC_AMINO_IDX_PAIRS)))
    ax.set_yticklabels(list(acids))
    ax.set_title(seq.id)

plt.show()
