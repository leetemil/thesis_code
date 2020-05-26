# First, command-line arguments
from args import get_vae_args
args = get_vae_args()

import time
from pathlib import Path
import torch
from torch import optim

import matplotlib.pyplot as plt

from models import VAE
from data import get_protein_dataloader, NUM_TOKENS, get_datasets
from training import train_epoch, validate, readable_time, get_memory_usage, mutation_effect_prediction, make_mutants
from visualization import plot_data, plot_loss, plot_spearman, plot_softmax, plot_protein_family_and_mutations, plot_tsne, plot_gaussian_distribution

if __name__ == "__main__" or __name__ == "__console__":
    # Argument postprocessing
    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device specified, but CUDA is not available. Use --device cpu.")
    device = torch.device(args.device)
    try:
        device_name = torch.cuda.get_device_name()
    except:
        device_name = "CPU"

    print(f"Using device: {device_name}")

    # Load data
    all_data, train_data, val_data = get_datasets(args.data, device, args.train_ratio, use_saved = True)

    # Construct dataloaders for batches
    train_loader = get_protein_dataloader(train_data, batch_size = args.batch_size, shuffle = True, random_weighted_sampling = args.random_weighted_sampling)
    val_loader = get_protein_dataloader(val_data, batch_size = args.batch_size)
    print("Data loaded!")

    # Define model and optimizer
    data_size = all_data[0][0].size(-1) * NUM_TOKENS
    model = VAE(
        [data_size] + args.layer_sizes + [data_size],
        NUM_TOKENS,
        z_samples = args.z_samples,
        dropout = args.dropout,
        use_bayesian = args.bayesian,
        use_param_loss = args.param_loss,
        use_dictionary = args.dictionary,
        label_smoothing = args.label_smoothing,
        warm_up = args.warm_up
    ).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.L2)

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
        print(f"Model loaded.")

    # Train, validate, save
    show = False
    save = False
    if args.visualize_style == "show" or args.visualize_style == "both":
        show = True
    if args.visualize_style == "save" or args.visualize_style == "both":
        save = True

    best_loss = float("inf")
    patience = args.patience
    try:
        epochs = []
        improved_epochs = []
        train_nll_losses = []
        train_kld_losses = []
        train_param_klds = []
        train_total_losses = []
        val_nll_losses = []
        val_kld_losses = []
        val_param_klds = []
        val_total_losses = []
        spearman_rhos = []

        subset_labels = True

        if args.visualize_interval != "never":
            plot_data(args.results_dir / Path(f"epoch_0_val_loss_inf.png") if save else None, args.figure_type, model, all_data, args.batch_size, show = show, only_subset_labels=subset_labels)
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(epoch = epoch, model = model, optimizer = optimizer, train_loader = train_loader, log_interval = args.log_interval, clip_grad_norm = args.clip_grad_norm, clip_grad_value = args.clip_grad_value, random_weighted_sampling = args.random_weighted_sampling)

            if args.val_ratio > 0:
                val_loss, val_metrics = validate(epoch, model, val_loader)
                loss_str = "Validation"
                loss_value_str = f"{val_loss:.5f}"
                val_str = f"{loss_str} loss: {loss_value_str} "
                val_nll_losses.append(val_metrics["nll_loss"])
                val_kld_losses.append(val_metrics["kld_loss"])
                val_param_klds.append(val_metrics["param_kld"])
                val_total_losses.append(val_loss)

                improved = val_loss < best_loss
            else:
                loss_str = "Training"
                loss_value_str = f"{train_loss:.5f}"
                val_str = ""
                improved = train_loss < best_loss

            epochs.append(epoch)
            train_nll_losses.append(train_metrics["nll_loss"])
            train_kld_losses.append(train_metrics["kld_loss"])
            train_param_klds.append(train_metrics["param_kld"])
            train_total_losses.append(train_loss)

            rho_str = ""
            if improved:
                # If model improved, save the model
                model.save(model_save_name)
                print(f"{loss_str} loss improved from {best_loss:.5f} to {loss_value_str}. Saved model to: {model_save_name}")
                best_loss = val_loss if args.val_ratio > 0 else train_loss
                patience = args.patience
                improved_epochs.append(epoch)

            if args.visualize_interval == "always" or (args.visualize_interval == "improvement" and improved):
                with torch.no_grad():
                    rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count_training, args.results_dir, savefig = False)

                    spearman_rhos.append(rho)
                    rho_str = f" Spearman's Rho: {rho:.3f}"

                name = args.results_dir / Path(f"epoch_{epoch}_loss_{loss_value_str}.png") if save else None
                plot_data(name, args.figure_type, model, all_data, rho, args.batch_size, show = show, only_subset_labels = subset_labels)

            elif args.patience:
                # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                patience -= 1
                if patience == 0:
                    print(f"Model has not improved for {args.patience} epochs. Stopping training. Best {loss_str.lower()} loss achieved was: {best_loss:.5f}.")
                    break

            if args.visualize_interval == "always":
                plot_epochs = range(len(spearman_rhos))
            else:
                plot_epochs = improved_epochs

            spearman_name = args.results_dir / Path("spearman_rhos.png")
            plot_spearman(args.data, spearman_name, plot_epochs, spearman_rhos)

            train_name = args.results_dir / Path("Train_losses.png")
            plot_loss(epochs, train_nll_losses, train_kld_losses, train_param_klds, train_total_losses, val_nll_losses, val_kld_losses, val_param_klds, val_total_losses, train_name, figure_type = args.figure_type, show = show)

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} {val_str}Time: {readable_time(time.time() - start_time)}{rho_str} Memory: {get_memory_usage(device):.2f}GiB", end = "\n\n")

        print('Computing mutation effect prediction correlation...')
        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
                print("Model loaded.")
            rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, args.ensemble_count, args.results_dir)

            softmax_proteins, *_ = next(iter(train_loader))
            softmax_proteins = softmax_proteins[:4]
            predictions = model.get_predictions(softmax_proteins).permute(0, 2, 1).exp().cpu().numpy()
            softmax_name = args.results_dir / Path("softmax.png")
            plot_softmax(softmax_name, predictions)

            mutants_fn = make_mutants(args.query_protein, args.data_sheet, args.metric_column, device)
            mutant_data = next(mutants_fn)

            plot_tsne(args.results_dir / Path("tsne.png"), args.figure_type, model, all_data, rho, args.batch_size, show = show, only_subset_labels = subset_labels)
            breakpoint()

            plot_protein_family_and_mutations(
                model = model,
                protein_family_data = all_data,
                mutant_data = mutant_data,
                batch_size = args.batch_size,
                model_path = args.results_dir
            )


        print(f'Spearman\'s Rho: {rho}')

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best loss achieved was: {best_loss:.5f}.\n")
        breakpoint()
    finally:
        if args.visualize_interval == "always":
            plot_epochs = range(len(spearman_rhos))
        else:
            plot_epochs = improved_epochs

        spearman_name = args.results_dir / Path("spearman_rhos.png")
        plot_spearman(args.data, spearman_name, plot_epochs, spearman_rhos)

        train_name = args.results_dir / Path("Train_losses.png")
        plot_loss(epochs, train_nll_losses, train_kld_losses, train_param_klds, train_total_losses, val_nll_losses, val_kld_losses, val_param_klds, val_total_losses, train_name, figure_type = args.figure_type, show = show)
