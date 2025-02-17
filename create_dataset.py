import os
import argparse
from argparse import Namespace
from pprint import pprint
import datetime as dt
from tqdm import tqdm
from itertools import groupby
from pathlib import Path

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import numpy as np
from egs.streamingDSU import ASRDataset
import espnetez as ez
import logging
# logging.basicConfig(level=logging.INFO)


def get_time(date_fmt: str) -> str:
    return dt.datetime.now().strftime(date_fmt)


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference of unit-to-text model")

    # general arguments
    parser.add_argument('--config', type=Path, required=True, help="Path to the model configuration file")
    parser.add_argument('--ckpt', type=Path, required=True, help="Path to the checkpoint model.")
    parser.add_argument('--dump_dir', type=Path, default=Path("output"), help="Path to the output directory")
    parser.add_argument('--dataset_split', type=str, required=True, choices=("train", "dev", "test_clean", "test_other", "test_1h"))
    parser.add_argument('--create_bpe', action='store_true')
    args = parser.parse_args()

    assert args.config.exists()
    assert args.ckpt.exists()

    return args


if __name__ == '__main__':
    # Step 0. Parse command-line arguments and load configuration
    args = parse_args()
    config = OmegaConf.load(args.config)
    pprint(args)
    pprint(config)
    OmegaConf.register_new_resolver("now", get_time)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1. Initialize dataset and model
    dataset = ASRDataset(
        split=args.dataset_split,
        num_proc=4,
    )

    unit_model = instantiate(config.model)
    d = torch.load(args.ckpt)
    if 'layer_norm..mean' in d:
        d['layer_norm.mean'] = d.pop('layer_norm..mean')
        d['layer_norm.var'] = d.pop('layer_norm..var')
    unit_model.load_state_dict(d)
    unit_model.to(device)

    # Step 2. run inference and save results
    save_dir = args.dump_dir / args.dataset_split
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    with open(f"{save_dir}/units", "w") as f:
        for data in tqdm(dataset):
            audio = torch.from_numpy(data['audio']).to(device)

            # Forward pass
            with torch.no_grad():
                units = unit_model.inference(audio[None]).cpu().detach().numpy()

            unit_str = "".join([chr(int("4e00", 16) + int(c)) for c in units])
            f.write(f"{data['id']} {unit_str}\n")

    # Step 3. create BPE model
    if args.create_bpe:
        ez.preprocess.sentencepiece.train_sentencepiece(
            f"{save_dir}/units",
            output_path=f"{save_dir}/spm",
            vocab_size=3000,
        )
