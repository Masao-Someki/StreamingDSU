import argparse
from argparse import Namespace
from pprint import pprint

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import numpy as np

import espnetez as ez


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network for speech synthesis')

    # general arguments
    parser.add_argument('--expdir', type=str, default='./results', help='Path to save the trained model')
    parser.add_argument('--train_config', type=str, required=True, help='Path to the model configuration file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Step 0. Parse command-line arguments and load configuration
    args = parse_args()
    config = OmegaConf.load(args.train_config)
    pprint(args)
    pprint(config)

    # Step 1. Setup dataset
    data_info = {
        "speech": lambda x: x['audio'],
        "text": lambda x: np.array(x["units"]),
    }
    train_dataset = ez.dataset.ESPnetEZDataset(
        instantiate(config.train_dataset),
        data_info=data_info
    )
    dev_dataset = ez.dataset.ESPnetEZDataset(
        instantiate(config.dev_dataset),
        data_info=data_info
    )
    print(f'data_info: {data_info}')
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(dev_dataset)}')

    # Step 2. Setup model
    def build_model_fn(args):
        model = instantiate(config.model)
        return model

    # Step 3. Train the model
    # convert omegaconf to namespace
    config = Namespace(**OmegaConf.to_container(config))
    config.train['token_list'] = ["<unk>", "<s>", "</s>", "<pad>"]
    config.train['token_type'] = "char"
    trainer = ez.Trainer(
        task=config.task,
        train_config=config.train,
        train_dataset=train_dataset,
        valid_dataset=dev_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=args.expdir,
        stats_dir="stats/",
        ngpu=1
    )
    # trainer.collect_stats()
    trainer.train()

