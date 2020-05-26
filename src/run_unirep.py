from args import get_unirep_args
args = get_unirep_args()

import time
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import UniRep
from data import IterProteinDataset, get_variable_length_protein_dataLoader, NUM_TOKENS, IUPAC_SEQ2IDX
from training import train_batch, validate, readable_time, get_memory_usage
from visualization import plot_data

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
    train_data = IterProteinDataset(args.train_data, device = device)
    validation_data = IterProteinDataset(args.validation_data, device = device)
    val_len = len(validation_data)
    train_seqs_per_epoch = val_len * 9

    train_loader = get_variable_length_protein_dataLoader(train_data, batch_size = args.batch_size)
    val_loader = get_variable_length_protein_dataLoader(validation_data, batch_size = args.batch_size)
    print("Data loaded!")

    model = UniRep(NUM_TOKENS, IUPAC_SEQ2IDX["<pad>"], args.embed_size, args.hidden_size, args.num_layers).to(device)
    print(model.summary())
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.L2)

    if args.anneal_learning_rates:
        T_0 = 1
        T_mult = 2
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    else:
        scheduler = None

    model_save_name = args.results_dir / Path("model.torch")
    if model_save_name.exists():
        print(f"Loading saved model from {model_save_name}...")
        model.load_state_dict(torch.load(model_save_name, map_location = device)["state_dict"])
        print(f"Model loaded.")

    best_val_loss = float("inf")
    patience = args.patience
    seqs_processed = 0
    acc_train_loss = 0
    train_loss_count = 0
    prev_time = time.time()
    epoch = 0
    try:
        stop = False
        while not stop:
            total_batches = 39_069_211 // args.batch_size # used for annealing; maybe import number of data samples?
            for batch_idx, xb in enumerate(train_loader):
                batch_size, batch_train_loss, batch_metrics_dict = train_batch(model, optimizer, xb, args.clip_grad_norm, args.clip_grad_value, scheduler=scheduler, epoch=epoch, batch = batch_idx, num_batches=total_batches)
                seqs_processed += batch_size
                acc_train_loss += batch_train_loss
                train_loss_count += 1

                if seqs_processed >= train_seqs_per_epoch:
                    epoch += 1
                    train_loss = acc_train_loss / train_loss_count
                    seqs_processed = 0
                    acc_train_loss = 0
                    train_loss_count = 0
                    val_loss, _ = validate(epoch, model, val_loader)
                    print(f"Summary epoch: {epoch} Train loss: {train_loss:.5f} Validation loss: {val_loss:.5f} Time: {readable_time(time.time() - prev_time)} Memory: {get_memory_usage(device):.2f}GiB")
                    prev_time = time.time()

                    improved = val_loss < best_val_loss

                    if improved:
                        # If model improved, save the model
                        model.save(model_save_name)
                        print(f"Validation loss improved from {best_val_loss:.5f} to {val_loss:.5f}. Saved model to: {model_save_name}")
                        best_val_loss = val_loss
                        patience = args.patience
                    elif args.patience:
                        # If save path and patience was specified, and model has not improved, decrease patience and possibly stop
                        patience -= 1
                        if patience == 0:
                            print(f"Model has not improved for {args.patience} epochs. Stopping training. Best validation loss achieved was: {best_val_loss:.5f}.")
                            stop = True
                            break
                    print("")
                    if epoch >= args.epochs:
                        stop = True
                        break

    except KeyboardInterrupt:
        print(f"\n\nTraining stopped manually. Best validation loss achieved was: {best_val_loss:.5f}.\n")
        breakpoint()
