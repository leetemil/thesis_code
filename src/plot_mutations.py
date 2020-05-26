import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import VAE
from data import get_datasets, NUM_TOKENS, IUPAC_SEQ2IDX
from visualization import plot_mutations
from training import make_mutants

PICKLE_FILE = Path('data/files/mutation_data.pickle')

def protein_accuracy(trials, model, data):
    model.eval()
    print(f'{wt_id}: Prediction accuracies for {trials} proteins.')
    data = iter(data)
    for _ in range(trials):
        p, _,  p_seq = next(data)
        p_recon = model.reconstruct(p.unsqueeze(0)).squeeze(0).numpy()
        p = p.numpy()
        loss = 1 - (p == p_recon).mean()
        print(f'{p_seq.id:<60s}{100 * loss:>4.1f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Mutation representations", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    # Required arguments
    parser.add_argument("--model_path", type = Path, help = "Path to model directory.")
    parser.add_argument("--protein_family", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Protein family alignment data.")
    parser.add_argument("--data_sheet", type = str, default = "BLAT_ECOLX_Ranganathan2015", help = "Protein family data sheet in mutation_data.pickle.")
    parser.add_argument("--metric", type = str, default = "2500", help = "Metric column of sheet used for Spearman's Rho calculation.")
    parser.add_argument("-qp", "--query_protein", type = Path, default = Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file containing the query protein sequence for mutation effect prediction.")
    args = parser.parse_args()

    print("Arguments given:")
    for arg, value in args.__dict__.items():
        print(f"  {arg}: {value}")
    print("")

    # only tested on cpu device ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protein_dataset, *_ = get_datasets(args.protein_family, device, 1)
    print('Data loaded')

    wt, _, _, wt_seq = protein_dataset[0]
    wt_id = wt_seq.id
    size = len(wt) * NUM_TOKENS
    wt = wt.unsqueeze(0)

    model_file = args.model_path / Path("model.torch")

    model = VAE(
        [size, 1500, 1500, 2, 100, 2000, size],
        NUM_TOKENS,
        z_samples = 1,
        dropout = 0,
        use_bayesian = True,
        use_param_loss = True,#args.param_loss,
        use_dictionary = True,#args.dictionary,
        label_smoothing = 0,#args.label_smoothing,
        warm_up = 0#args.warm_up
    ).to(device)


    model.load_state_dict(torch.load(model_file, map_location=device)["state_dict"])

    mutants_fn = make_mutants(args.query_protein, args.data_sheet, args.metric, device)
    mutant_data = next(mutants_fn)

    with torch.no_grad():
        plot_mutations(model, mutant_data, args.model_path)
