import os
import argparse
from argparse import Namespace
from pprint import pprint
import datetime as dt
from tqdm import tqdm
from itertools import groupby

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import numpy as np

from espnet2.bin.mt_inference import Text2Text
from egs.finetuneUnit2Text import ASRDataset
from src.metrics import CER, WER
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
import logging
# logging.basicConfig(level=logging.INFO)


def get_time(date_fmt: str) -> str:
    return dt.datetime.now().strftime(date_fmt)


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference of unit-to-text model")

    # general arguments
    parser.add_argument('--config', type=str, required=True, help="Path to the model configuration file")
    parser.add_argument('--output_dir', type=str, default="output", help="Path to the output directory")
    parser.add_argument('--unit_dir', type=str)
    parser.add_argument('--split', type=str, default="test_clean")
    parser.add_argument('--beam_size', type=int, default=5, help="Beam size for inference")
    parser.add_argument('--mt_config', type=str, default="exp/wavlm_baseline/config.yaml", help="Path to the MT model configuration file")
    parser.add_argument('--mt_model', type=str, default="exp/wavlm_baseline/valid.acc.ave_10best.pth", help="Path to the MT model checkpoint")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Step 0. Parse command-line arguments and load configuration
    args = parse_args()
    config = OmegaConf.load(args.config)
    pprint(args)
    pprint(config)
    OmegaConf.register_new_resolver("now", get_time)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ASRDataset(
        split=args.split,
        src_bpe_path=config.dev_dataset.src_bpe_path,
        src_token_list_path=config.dev_dataset.src_token_list_path,
        tgt_bpe_path=config.dev_dataset.tgt_bpe_path,
        tgt_token_list_path=config.dev_dataset.tgt_token_list_path,
        num_proc=4,
        unit_path=f"{args.unit_dir}/{args.split}/units",
    )

    mt_model = Text2Text(
        mt_train_config=args.mt_config,
        mt_model_file=args.mt_model,
        beam_size=args.beam_size,
        ctc_weight=0.0,
        lm_weight=0.0,
        device=device
    )

    tokenizer = dataset.tokenizers["src"]
    converter = dataset.converters["src"]

    tgt_tokenizer = dataset.tokenizers["tgt"]
    tgt_converter = dataset.converters["tgt"]

    # Steo 2. Setup directories
    # eval_dir = f"{args.expdir}/eval_results/{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    eval_dir = f"{args.output_dir}/eval_results/{args.split}/{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    id2units = {}
    metrics_class = {
        "wer": WER(),
        "cer": CER(),
    }
    gts = []
    hyps = []
    ids = []
    for data in tqdm(dataset):
        units = data["units"]
        results = mt_model(units)
        text = tgt_tokenizer.tokens2text(tgt_converter.ids2tokens(data["text"]))

        ids.append(data["id"])
        gts.append(text)
        hyps.append(results[0][0])

    with open(os.path.join(eval_dir, "hyp.txt"), "w") as f:
        for aid, hyp in zip(ids, hyps):
            f.write(f"{aid}\t{hyp}\n")
    
    with open(os.path.join(eval_dir, "gt.txt"), "w") as f:
        for aid, gt in zip(ids, gts):
            f.write(f"{aid}\t{gt}\n")

    for k, metric in metrics_class.items():
        print(f"Processing {k}...")
        metrics_gts =  [metric.clean(gt)  for gt  in gts]
        metrics_hyps = [metric.clean(hyp) for hyp in hyps]

        metric.compute_and_save(metrics_gts, metrics_hyps, eval_dir)
