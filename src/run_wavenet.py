from args import get_wavenet_args
args = get_wavenet_args()

import time
from pathlib import Path

import torch
from torch import optim

from models import WaveNet
from data import VariableLengthProteinDataset, get_variable_length_protein_dataLoader, NUM_TOKENS, IUPAC_SEQ2IDX
from training import train_epoch, validate, readable_time, get_memory_usage, mutation_effect_prediction
from visualization import plot_spearman, plot_learning_rates, plot_softmax

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
    all_data = VariableLengthProteinDataset(args.data, device = device, max_len = 1024, use_weights = args.use_weights)
    train_length = int(len(all_data) * args.train_ratio)
    val_length = len(all_data) - train_length

    if args.validation_split_seed is not None:
        torch.manual_seed(args.validation_split_seed)
    train_data, val_data = torch.utils.data.random_split(all_data, [train_length, val_length])

    if args.seed is not None:
        torch.manual_seed(args.seed)
    elif args.validation_split_seed is not None:
        torch.manual_seed(torch.initial_seed())

    train_loader = get_variable_length_protein_dataLoader(train_data, batch_size = args.batch_size, shuffle = True, use_weights = args.use_weights)
    val_loader = get_variable_length_protein_dataLoader(val_data, batch_size = args.batch_size, use_weights = args.use_weights)

    print("Data loaded!")

    model = WaveNet(
        input_channels = NUM_TOKENS,
        residual_channels = args.residual_channels,
        out_channels = NUM_TOKENS,
        stacks = args.stacks,
        layers_per_stack = args.layers,
        total_samples = train_length,
        l2_lambda = args.L2,
		bias = args.bias,
        dropout = args.dropout,
        use_bayesian = args.bayesian,
        backwards = args.backwards
	).to(device)

    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)#, weight_decay = args.L2)

    if args.anneal_learning_rates:
        T_0 = 1 # Emil: I just picked a small number, no clue if any good
        T_mult = 2
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    else:
        scheduler = None

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
        print(f"Model loaded.")

    best_loss = float("inf")
    ensemble_count = args.ensemble_count if args.bayesian else 0
    patience = args.patience
    improved_epochs = []
    spearman_rhos = []
    spearman_name = args.results_dir / Path("spearman_rhos.png")

    # pick 4 random protein sequences
    if args.use_weights:
        softmax_proteins, *_ = next(iter(train_loader))
        softmax_proteins = softmax_proteins[:4]
    else:
        softmax_proteins = next(iter(train_loader))[:4]
    softmax_name = args.results_dir / Path("softmax.png")

    if args.anneal_learning_rates and args.plot_learning_rates:
        learning_rates = []
        learning_rates_name = args.results_dir / Path("learning_rates.png")

    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(epoch, model, optimizer, train_loader, args.log_interval, args.clip_grad_norm, args.clip_grad_value, scheduler)

            if args.val_ratio > 0:
                val_loss, val_metrics = validate(epoch, model, val_loader)
                loss_str = "Validation"
                loss_value_str = f"{val_loss:.5f}"
                val_str = f"{loss_str} loss: {loss_value_str} "
                improved = val_loss < best_loss

            else:
                loss_str = "Training"
                loss_value_str = f"{train_loss:.5f}"
                val_str = ""
                improved = train_loss < best_loss

            if improved:
                # If model improved, save the model
                model.save(model_save_name)
                print(f"{loss_str} loss improved from {best_loss:.5f} to {loss_value_str}. Saved model to: {model_save_name}")
                best_loss = val_loss if args.val_ratio > 0 else train_loss
                patience = args.patience

                with torch.no_grad():
                    rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, 100, args.results_dir, savefig = False)
                    predictions = model.get_predictions(softmax_proteins).permute(0, 2, 1).exp().cpu().numpy()

                spearman_rhos.append(rho)
                improved_epochs.append(epoch)
                plot_spearman(args.data, spearman_name, improved_epochs, spearman_rhos)
                plot_softmax(softmax_name, predictions)

            elif args.patience:
                # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                patience -= 1
                if patience == 0:
                    print(f"Model has not improved for {args.patience} epochs. Stopping training. Best {loss_str.lower()} loss achieved was: {best_loss:.5f}.")
                    break

            if args.plot_learning_rates and args.anneal_learning_rates:
                learning_rates += train_metrics['learning_rates']
                plot_learning_rates(learning_rates_name, learning_rates)

            print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} {val_str}Time: {readable_time(time.time() - start_time)} Memory: {get_memory_usage(device):.2f}GiB", end = "\n" if improved else "\n\n")

            if improved:
                print(f"Spearman\'s Rho: {rho:.3f}", end = "\n\n")

        print('Computing mutation effect prediction correlation...')

        with torch.no_grad():
            if model_save_name.exists():
                model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])

            predictions = model.get_predictions(softmax_proteins).permute(0,2,1).exp().cpu().numpy()
            plot_softmax(softmax_name, predictions)

            rho = mutation_effect_prediction(model, args.data, args.query_protein, args.data_sheet, args.metric_column, device, ensemble_count, args.results_dir)

        print(f'Spearman\'s Rho: {rho:.3f}')

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_loss:.5f}.\n")
        breakpoint()
