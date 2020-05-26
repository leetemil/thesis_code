import torch
import argparse
from pathlib import Path

args = argparse.ArgumentParser()
args.add_argument("input_files", type = Path, nargs = "*")
config = args.parse_args()

for input_file in config.input_files:
    old_dict = torch.load(input_file)["state_dict"]

    old_keys = old_dict.keys()
    new_keys = map(lambda ok: ok.replace("module.", "inner_model.") if ok.startswith("module.") else "inner_model." + ok, old_keys)

    keys = zip(new_keys, old_keys)
    new_dict = {nk: old_dict[ok] for nk, ok in keys}
    torch.save(new_dict, input_file.with_suffix(".tape"))
