from args import get_evaluate_ensemble_args
args = get_evaluate_ensemble_args()

from pathlib import Path
import glob
import pickle
from Bio import SeqIO
from data import get_datasets, NUM_TOKENS, IUPAC_SEQ2IDX, IUPAC_IDX2SEQ, seq2idx, idx2seq


import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from models import VAE, UniRep, WaveNet, LossTransformer
from training import mutation_effect_prediction

# Device
if args.device == "cuda" and not torch.cuda.is_available():
    raise ValueError("CUDA device specified, but CUDA is not available. Use --device cpu.")
device = torch.device(args.device)
try:
    device_name = torch.cuda.get_device_name()
except:
    device_name = "CPU"

print(f"Using device: {device_name}")

# Construct models
models = []
for directory in args.model_directories:
    model_list = glob.glob(str(directory / Path("*.torch")))
    for model_path in model_list:
        load_dict = torch.load(model_path, map_location = device)

        name = load_dict["name"]
        state_dict = load_dict["state_dict"]
        args_dict = load_dict["args_dict"]

        if name == "VAE":
            model_type = VAE
        elif name == "UniRep":
            model_type = UniRep
        elif name == "WaveNet":
            model_type = WaveNet
        elif name == "Transformer":
            model_type = LossTransformer
        else:
            raise ValueError("Unrecognized model name.")

        # args_dict["use_bayesian"] = args_dict.pop("bayesian") # use this if you get bayesian keyword error for wavenet
        model = model_type(**args_dict).to(device)
        model.load_state_dict(state_dict)

        models.append(model)

with torch.no_grad():
    # Evaluate on mutation effect
    # scores = mutation_effect_prediction(models[0], args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir, return_scores = True)

    # acc_m_logp = 0
    # acc_wt_logp = 0
    # for model in models:
    #     m_logp, wt_logp = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir, return_logps = True)

    #     acc_m_logp += m_logp
    #     acc_wt_logp += wt_logp

    # ensemble_m_logp = acc_m_logp / len(models)
    # ensemble_wt_logp = acc_wt_logp / len(models)

    # predictions = ensemble_m_logp - ensemble_wt_logp

    # plt.scatter(predictions.cpu(), scores)
    # plt.title("Correlation")
    # plt.xlabel("$\\Delta$-elbo")
    # plt.ylabel("Experimental value")
    # plt.savefig(args.results_dir / Path("Correlation_scatter.png"))

    # cor, _ = spearmanr(scores, predictions.cpu())
    # print(f'Ensemble Spearman\'s Rho over {len(models)} models: {cor:5.3f}')

    if args.accuracy and isinstance(model, VAE):

        PICKLE_FILE = Path('data/files/mutation_data.pickle')
        with open(PICKLE_FILE, 'rb') as f:
            proteins = pickle.load(f)
            p = proteins[args.data_sheet].dropna(subset=['mutation_effect_prediction_vae_ensemble']).reset_index(drop=True)

        sequences = SeqIO.parse(args.data, "fasta")
        wt_seq = next(sequences)
        other_seq = next(sequences)

        chosen_seq = wt_seq

        wt = seq2idx(chosen_seq, device)
        batch_wt = wt.unsqueeze(0)

        predictions = []

        for model in models:
            softmax = model.get_predictions(batch_wt).permute(0, 2, 1)

            for i in range(args.ensemble_count - 1):
                softmax += model.get_predictions(batch_wt).permute(0, 2, 1)
                # encoded = model.encode(wt.unsqueeze(0)).mean
                # decoded = model.sample(encoded)
            predictions += [softmax / args.ensemble_count]

        reconstructed = torch.cat(predictions).mean(0).exp().argmax(dim = -1)

        in_seq = str(chosen_seq.seq).upper()
        out_seq = idx2seq(reconstructed.squeeze().cpu().numpy()).replace('<mask>', '.')

        accuracy = sum(a1 == a2 for (a1, a2) in zip(in_seq, out_seq)) / len(in_seq)

        # print(f'Encoded representation (mean of encoder distribution):\n{encoded.squeeze()}\n')
        print(f'{in_seq}')
        print(f'{out_seq}')
        print(accuracy)
