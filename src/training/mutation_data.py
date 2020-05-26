import argparse
from pathlib import Path

from datetime import datetime
import pickle
import math

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from Bio import SeqIO

from models import VAE, WaveNet
from data import get_datasets, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq

PICKLE_FILE = Path('data/files/mutation_data.pickle')

# def protein_accuracy(trials = 100, model = model, data = protein_dataset):
#     model.eval()
#     print(f'{wt_id}: Prediction accuracies for {trials} proteins.')
#     data = iter(data)
#     for _ in range(trials):
#         p, _,  p_seq = next(data)
#         p_recon = model.reconstruct(p.unsqueeze(0)).squeeze(0).numpy()
#         p = p.numpy()
#         loss = 1 - (p == p_recon).mean()
#         print(f'{p_seq.id:<60s}{100 * loss:>4.1f}%')

def make_mutants(data_path, sheet, metric_column, device):
    print("Making mutants...")
    # load mutation and experimental pickle
    with open(PICKLE_FILE, 'rb') as f:
        proteins = pickle.load(f)
        p = proteins[sheet].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

    # load dataset
    wt_seq = next(SeqIO.parse(data_path, "fasta"))
    wt_indices = np.array([i for i, c in enumerate(str(wt_seq.seq))])
    wt = seq2idx(wt_seq, device)

    offset = int(wt_seq.id.split("/")[1].split("-")[0])

    positions = wt_indices + offset
    positions_dict = {pos: i for i, pos in enumerate(positions)}

    mutants_list, scores = zip(*list(filter(lambda t: not math.isnan(t[1]), zip(p.mutant, p[metric_column]))))
    mutants_list, scores = list(mutants_list), list(scores)
    wt_present = mutants_list[-1].lower() == "wt"
    if wt_present:
        del mutants_list[-1]
        del scores[-1]

    data_size = len(mutants_list)
    mutants = wt.repeat(data_size, 1)

    for i, position_mutations in enumerate(mutants_list):
        mutations = position_mutations.split(":")
        for mutation in mutations:
            wildtype = IUPAC_SEQ2IDX[mutation[0]]
            mutant = IUPAC_SEQ2IDX[mutation[-1]]

            # handle special offset case
            if sheet == "parEparD_Laub2015_all":
                offset = 103
            else:
                offset = 0

            location = positions_dict[int(mutation[1:-1]) + offset]

            assert mutants[i, location] == wildtype, f"{IUPAC_IDX2SEQ[mutants[i, location].item()]}, {IUPAC_IDX2SEQ[wildtype]}, {location}, {i}"
            mutants[i, location] = mutant

    while True:
        yield mutants, wt, scores

def get_elbos(model, wt, mutants, ensemble_count):
    if isinstance(model, VAE):
        acc_m_elbo = 0
        acc_wt_elbo = 0

        batch = torch.cat([wt.unsqueeze(0), mutants])

        for i in range(ensemble_count):
            elbos, *_ = model.protein_logp(batch)
            wt_elbo = elbos[0]
            m_elbo = elbos[1:]
            acc_m_elbo += m_elbo
            acc_wt_elbo += wt_elbo

        print("Done!" + " " * 50, end = "\r")

        mutants_logp = acc_m_elbo / ensemble_count
        wt_logp = acc_wt_elbo / ensemble_count

    else:
        wt_pad = F.pad(wt, (1, 0), value = IUPAC_SEQ2IDX["<cls>"])
        wt_pad = F.pad(wt_pad, (0, 1), value = IUPAC_SEQ2IDX["<sep>"])
        wt_logp = model.protein_logp(wt_pad.unsqueeze(0))

        mutants = F.pad(mutants, (1, 0), value = IUPAC_SEQ2IDX["<cls>"])
        mutants = F.pad(mutants, (0, 1), value = IUPAC_SEQ2IDX["<sep>"])

        batch_size = 2048
        batches = len(mutants) // batch_size + 1

        model_logps = []

        ensemble_count = ensemble_count if isinstance(model, WaveNet) and model.bayesian else 1

        if isinstance(model, WaveNet) and model.bayesian:
            print('Emil: Warning, you are using wavenet bayesian. the ensemble count currently does not calculate logp wt from ensemble. You should probaby fix this.')
            breakpoint()

        for m in range(ensemble_count):
            log_probs = []
            for i in range(batches):
                batch_mutants = mutants[batch_size * i: batch_size * (i + 1)]
                m_logp = model.protein_logp(batch_mutants)
                log_probs.append(m_logp)

            model_logps.append(torch.cat(log_probs))

        if ensemble_count > 1:
            print("Done!" + " " * 50)

        mutants_logp = sum(model_logps) / ensemble_count

    return mutants_logp, wt_logp

mutants_fn = None

def mutation_effect_prediction(model, data_path, query_protein, sheet, metric_column, device, ensemble_count = 500, results_dir = Path("."), savefig = True, return_scores = False, return_logps = False):
    model.eval()
    global mutants_fn
    if mutants_fn is None:
        mutants_fn = make_mutants(query_protein, sheet, metric_column, device)

    mutants, wt, scores = next(mutants_fn)
    if return_scores:
        return scores

    mutants_logp, wt_logp = get_elbos(model, wt, mutants, ensemble_count)
    if return_logps:
        return mutants_logp, wt_logp

    predictions = mutants_logp - wt_logp

    if savefig:
        plt.scatter(predictions.cpu(), scores)
        plt.title("Correlation")
        plt.xlabel("$\\Delta$-elbo")
        plt.ylabel("Experimental value")
        plt.savefig(results_dir / Path("Correlation_scatter.png"))

    cor, _ = spearmanr(scores, predictions.cpu())
    return abs(cor) # we only care about absolute value

if __name__ in ["__main__", "__console__"]:
    parser = argparse.ArgumentParser(description = "Mutation prediction and analysis", formatter_class = argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    # Required arguments
    parser.add_argument("--data", type = Path, default = Path("data/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m"), help = "Fasta input file of sequences.")
    parser.add_argument("--data_sheet", type = str, default = "BLAT_ECOLX_Ranganathan2015", help = "Protein family data sheet in mutation_data.pickle.")
    parser.add_argument("--metric_column", type = str, default = "2500", help = "Metric column of sheet used for Spearman's Rho calculation.")
    parser.add_argument("--ensemble_count", type = int, default = 2000, help = "How many samples of the model to use for evaluation as an ensemble.")
    parser.add_argument("--results_dir", type = Path, default = Path(f"results_{datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}"), help = "Directory to save results to.")

    with torch.no_grad():
        args = parser.parse_args()

        print("Arguments given:")
        for arg, value in args.__dict__.items():
            print(f"  {arg}: {value}")
        print("")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        protein_dataset, *_ = get_datasets(args.data, device, 0.8)
        print('Data loaded')

        wt, *_ = protein_dataset[0]
        size = len(wt) * NUM_TOKENS

        # load model
        model = VAE([size, 1500, 1500, 30, 100, 2000, size], NUM_TOKENS, use_dictionary = False).to(device)

        try:
            model.load_state_dict(torch.load(args.results_dir / Path("model.torch"), map_location=device)["state_dict"])
        except FileNotFoundError:
            pass

        cor = mutation_effect_prediction(model, args.data, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir)

        print(f'Spearman\'s Rho: {cor:5.3f}')
