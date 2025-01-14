import argparse
from argparse import Namespace
from pprint import pprint
from tqdm import tqdm

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import numpy as np

from egs.streamingDSU import ASRDataset


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network for speech synthesis')

    # general arguments
    parser.add_argument('--split', type=str, default='test_clean', help='Path to save the trained model')
    parser.add_argument('--ckpt', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to the model configuration file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Step 0. Parse command-line arguments and load configuration
    args = parse_args()
    config = OmegaConf.load(args.config)
    pprint(args)
    pprint(config)

    # Step 1. Setup dataset
    dataset = ASRDataset(
        split=args.split,
        num_proc=4,
    )
    model = instantiate(config.model)
    d = torch.load(args.ckpt)
    model.load_state_dict(d)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    id2units = {}
    for data in tqdm(dataset):
        audio = torch.from_numpy(data['audio']).to(device)

        # Forward pass
        with torch.no_grad():
            units = model.inference(audio[None])

        id2units[data['id']] = units.cpu().detach().tolist()

    config_name = args.config.split("/")[-1].split(".")[0]
    with open(f"{args.split}_units_{config_name}.txt", "w") as f:
        for aid, units in id2units.items():
            f.write(f"{aid}\t{' '.join(map(str, units))}\n")
        