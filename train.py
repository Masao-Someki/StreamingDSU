import argparse
from argparse import Namespace
from pprint import pprint
from datetime import datetime
import os

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import numpy as np

import espnetez as ez

import logging

logging.basicConfig(level=logging.INFO)


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network predicting discrete units")

    # general arguments
    parser.add_argument('--train_config', type=str, required=True, help="Path to the model configuration file")

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
    config.train['drop_last_iter'] = True
    config.train['shuffle_within_batch'] = False

    config_name = os.path.basename(args.train_config).split(".")[0]
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    expdir = f"exp/{config_name}/{current_date}"
    
    default_config = ez.get_ez_task(config.task).get_default_config()
    default_config.update(config.train)

    trainer = ez.Trainer(
        task=config.task,
        train_config=default_config,
        train_dataset=train_dataset,
        valid_dataset=dev_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=expdir,
        stats_dir="stats/",
        ngpu=1
    )
    trainer.collect_stats()
    trainer.train()

