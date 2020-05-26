from collections import defaultdict
from pathlib import Path
import itertools

import torch
import numpy as np
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import VAE
from training import make_mutants
from data import get_protein_dataloader, IUPAC_AMINO_IDX_PAIRS

def get_pfam_label_dict():
    file = Path("data/files/PF00144_full_length_sequences_labeled.fasta")
    seqs = SeqIO.parse(file, "fasta")

    def getKeyLabel(seq):
        s = seq.description.split(" ")
        return s[0], s[2].replace("[", "").replace("]", "")

    return {protein_id: label for protein_id, label in map(getKeyLabel, seqs)}

PFAM_LABEL_DICT = get_pfam_label_dict()

def get_BLAT_label_dict(file):
    with open(file, "r") as f:
        lines = f.readlines()

    return dict([line[:-1].split(": ") for line in lines])

BLAT_LABEL_DICT = get_BLAT_label_dict(Path("data/files/alignments/BLAT_ECOLX_1_b0.5_LABELS.a2m"))

BLAT_HMMERBIT_LABEL_DICT = get_BLAT_label_dict(Path("data/files/alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_LABELS.a2m"))

def preprocess_data(model, dataset, batch_size, only_subset_labels = True):
    subset_labels = set([
        "Acidobacteria",
        "Actinobacteria",
        "Bacteroidetes",
        "Chloroflexi",
        "Cyanobacteria",
        "Deinococcus-Thermus",
        "Other",
        "Firmicutes",
        "Fusobacteria",
        "Proteobacteria"
    ])

    dataloader = get_protein_dataloader(dataset, batch_size = batch_size, get_seqs = True)

    scatter_dict = defaultdict(lambda: [])
    with torch.no_grad():
        for xb, weights, neff, seqs in dataloader:
            ids = [s.id for s in seqs]
            dist = model.encode(xb)
            mean = dist.mean.cpu()
            for point, ID in zip(mean, ids):
                try:
                    label = BLAT_HMMERBIT_LABEL_DICT[ID]
                    if only_subset_labels:
                        if label in subset_labels:
                            scatter_dict[label].append(point)
                    else:
                        scatter_dict[label].append(point)
                except KeyError:
                    if not only_subset_labels:
                        scatter_dict["Others"].append(point)

    all_points_list = list(itertools.chain(*scatter_dict.values()))
    all_points = torch.stack(all_points_list) if len(all_points_list) > 0 else torch.zeros(0, 0)

    return scatter_dict, all_points

def plot_tsne(name, figure_type, model, dataset, rho, batch_size = 64, only_subset_labels = True, show = False, tsne_dim = 2):

    scatter_dict, all_points = preprocess_data(model, dataset, batch_size = batch_size, only_subset_labels = only_subset_labels)

    tsne_fig = plt.figure(figsize = [10.0, 10.0])

    if tsne_dim != 2:
        raise NotImplementedError("Only supports t-SNE dimensionality reduction to 2d.")

    if all_points.size(1) > 2:
        tsne = TSNE(tsne_dim, perplexity = 50)
        transformed_points = tsne.fit_transform(all_points)

    else:
        transformed_points = all_points

    size = 5
    count = 0
    for label, points in scatter_dict.items():
        l = len(points)
        points = transformed_points[count:count + l]
        count += l
        plt.scatter(points[:, 0], points[:, 1], s = size, label = label)

    plt.legend(bbox_to_anchor=(1.04, 1), markerscale = 6, fontsize = 14)
    plt.tight_layout()

    if name is not None:
        tsne_fig.savefig(name.with_suffix(figure_type), bbox_inches='tight')

    if show:
        plt.show()

    plt.close(tsne_fig)

def plot_data(name, figure_type, model, dataset, rho, batch_size = 64, only_subset_labels = True, show = False, pca_dim = 2):

    scatter_dict, all_points = preprocess_data(model, dataset, batch_size = batch_size, only_subset_labels = only_subset_labels)

    pca_fig = plt.figure(figsize = [10.0, 10.0])
    plt.xlabel("$z_1$", fontsize = 16)
    plt.ylabel("$z_2$", fontsize = 16)

    if all_points.size(1) > 2:
        if pca_dim == 3:
            axis = Axes3D(pca_fig)
        pca = PCA(pca_dim)
        pca.fit(all_points)
        explained_variance = pca.explained_variance_ratio_.sum()
        plt.title(f"PCA of encoded points ({explained_variance:.3f} explained variance). Spearman's $\\rho$: {rho:.3f}")

        # Make explained variance figure
        variance_fig = plt.figure(figsize = [10, 3])
        plt.title("Explained variance of principal components")
        plt.xlabel("Principal components")
        plt.ylabel("Ratio of variance")
        # plt.ylim((0, 1))
        plt.ylim((0, 0.2))

        pca_highdim = PCA(all_points.size(1))
        pca_highdim.fit(all_points)
        explained_variances = pca_highdim.explained_variance_ratio_
        plt.plot(range(len(explained_variances)), explained_variances, label = "Explained variance ratio")

        if name is not None:
            plt.tight_layout()
            variance_fig.savefig(name.with_name("explained_variance").with_suffix(figure_type), bbox_inces = 'tight')
        plt.close(variance_fig)
        plt.figure(pca_fig.number)

    size = 5
    for label, points in scatter_dict.items():
        points = torch.stack(points)
        if points.size(1) == 2:
            plt.scatter(points[:, 0], points[:, 1], s = size, label = label)
        elif points.size(1) > 2:
            pca_points = pca.transform(points)
            if pca_dim == 2:
                plt.scatter(pca_points[:, 0], pca_points[:, 1], s = size, label = label)
            elif pca_dim == 3:
                axis.scatter(pca_points[:, 0], pca_points[:, 1], pca_points[:, 2], s = size, label = label)

    plt.legend(bbox_to_anchor=(1.04, 1), markerscale = 6, fontsize = 14)

    if name is not None:
        pca_fig.savefig(name.with_suffix(figure_type), bbox_inches='tight')

    if show:
        plt.show()

    plt.close(pca_fig)

def plot_spearman(data, name, epochs, rhos):
    fig = plt.figure()
    plt.title(data.stem.split('_')[0])
    plt.xlabel('Epochs')
    plt.ylabel('Spearman\'s $\\rho$')
    plt.plot(epochs, rhos, '+--')
    plt.savefig(name)
    plt.close(fig)

def plot_learning_rates(name, learning_rates):
    fig = plt.figure()
    plt.title('Annealed Learning Rates')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.plot(learning_rates)
    plt.savefig(name)
    plt.close(fig)

def plot_softmax(name, predictions):
    fig, axes = plt.subplots(len(predictions), figsize = (25, 15))
    for ax, prediction in zip(axes, predictions):
        ax.imshow(prediction.T, cmap=plt.get_cmap("Blues"))
        acids, _ = zip(*IUPAC_AMINO_IDX_PAIRS)
        ax.set_yticks(np.arange(len(IUPAC_AMINO_IDX_PAIRS)))
        ax.set_yticklabels(list(acids))
        # ax.set_title(seq.id)

    plt.tight_layout()
    plt.savefig(name, bbox_inces = 'tight')
    plt.close(fig)

def plot_loss(epochs, train_recon_loss, train_kld_loss, train_param_loss, train_total_loss, val_recon_loss, val_kld_loss, val_param_loss, val_total_loss, name, figure_type = 'png', show = False, logscale = True):

    fig, axs = plt.subplots(2, 2, figsize = (12, 7))

    if logscale:
        axs[0, 1].set(yscale = 'log')
        axs[1, 1].set(yscale = 'log')

    axs[0, 0].plot(epochs, train_recon_loss, label = "Train")
    axs[0, 0].set_title('Reconstruction Loss')
    axs[0, 0].set(ylabel='Loss')
    axs[0, 1].plot(epochs, train_param_loss, label = "Train")
    axs[0, 1].set_title('$\\theta$ loss')
    axs[0, 1].set(ylabel='Loss')
    axs[1, 0].plot(epochs, train_kld_loss, label = "Train")
    axs[1, 0].set_title('KLD loss')
    axs[1, 0].set(xlabel='Epoch', ylabel='Loss')
    axs[1, 1].plot(epochs, train_total_loss, label = "Train")
    axs[1, 1].set_title('Total loss')
    axs[1, 1].set(xlabel='Epoch', ylabel='Loss')

    if len(val_recon_loss) > 0:
        axs[0, 0].plot(epochs, val_recon_loss, label = "Validation")
        axs[0, 1].plot(epochs, val_param_loss, label = "Validation")
        axs[1, 0].plot(epochs, val_kld_loss, label = "Validation")
        axs[1, 1].plot(epochs, val_total_loss, label = "Validation")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    if name is not None:
        plt.savefig(name.with_suffix(figure_type))

    if show:
        plt.show()

    plt.close(fig)

def make_latent_mutations(model, mutant_data):
    mutants, wt, scores = mutant_data

    model.eval()
    mutants_wt = torch.cat([mutants, wt.unsqueeze(0)])
    z_mutants = model.get_representation(mutants_wt).cpu().numpy()
    z_wt = model.get_representation(wt.unsqueeze(0)).cpu().numpy()

    return z_mutants, z_wt

def plot_mutations(model, mutant_data, model_path):
    z_mutants, z_wt = make_latent_mutations(model, mutant_data)

    if z_mutants.shape[1] > 2:
        print(f'Latent space has {z_mutants.shape[1]} dimensions. Using PCA to project to 2d.')
        pca = PCA(2)
        z_mutants = pca.fit_transform(z_mutants)
        z_wt = pca.transform(z_wt)

    plt.figure(figsize = [10.0, 10.0])
    plt.scatter(z_mutants[:, 0], z_mutants[:, 1], s = 5, label = 'Mutants')
    plt.scatter(z_wt[:, 0], z_wt[:, 1], s = 30, c = 'black', label = 'Wild Type')
    plt.legend(markerscale = 2, fontsize = 14)
    plt.savefig(model_path / Path("mutation_plot.png"), bbox_inces = 'tight')
    plt.show()

def plot_protein_family_and_mutations(model, protein_family_data, mutant_data, batch_size, model_path):
    z_mutants, z_wt = make_latent_mutations(model, mutant_data)
    scatter_dict, all_points = preprocess_data(model, protein_family_data, batch_size = batch_size)

    fig, ax = plt.subplots(figsize=[14, 10])

    size = 3
    mutant_color = 'seagreen'
    for label, points in scatter_dict.items():
        points = torch.stack(points)

        if points.size(1) != 2:
            raise NotImplementedError('Currently only 2d representations supported.')

        plt.scatter(points[:, 0], points[:, 1], s = size, label = label)

    # plt.legend(bbox_to_anchor=(1.04, 1), markerscale = 6, fontsize = 14)
    plt.scatter(z_mutants[:, 0], z_mutants[:, 1], s = size, c = mutant_color, label = 'Mutants')
    plt.scatter(z_wt[:, 0], z_wt[:, 1], s = 30, c = 'black', label = 'Wild Type')

    # zoomed plot
    axins = ax.inset_axes([1.04, 0.0, 4/14, 4/10])

    for label, points in scatter_dict.items():
        points = torch.stack(points)

        if points.size(1) != 2:
            raise NotImplementedError('Currently only 2d representations supported.')

        axins.scatter(points[:, 0], points[:, 1], s = size, label = label)

    axins.scatter(z_mutants[:, 0], z_mutants[:, 1], s = size - 2, c = mutant_color, label = 'Mutants')
    axins.scatter(z_wt[:, 0], z_wt[:, 1], s = 30, c = 'black', label = 'Wild Type')



    # sub region of the original image
    x1, x2, y1, y2 = -0.16, -0.115, -0.58, -0.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    ax.indicate_inset_zoom(axins, label='_nolegend_')
    lgnd = plt.legend(bbox_to_anchor=(1.04, 1), markerscale = 6, fontsize = 14)

    # lgnd.legendHandles = lgnd.legendHandles[1:]

    for handle in lgnd.legendHandles:
        handle.set_sizes([100.0])

    plt.tight_layout()
    plt.savefig(model_path / Path("family_and_mutations.png"), bbox_inces = 'tight')
    plt.show()
    return

def plot_gaussian_distribution(mean, logvar, fig):
    std = (0.5 * logvar).exp()
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    fig.plot(x, stats.norm.pdf(x, mean, std))

# if __name__ == "__main__":
#     device = torch.device("cuda")

#     model = VAE([2594, 128, 2]).to(device)
#     model.load_state_dict(torch.load("model.torch")["state_dict"])

#     plot_data(None, model, "data/PF00144_full.txt", device)
