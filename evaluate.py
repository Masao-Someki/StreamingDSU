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
from egs.streamingDSU import ASRDataset
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
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the checkpoint model.")
    parser.add_argument('--output_dir', type=str, default="output", help="Path to the output directory")
    parser.add_argument('--dataset_split', type=str, choices=("test_clean", "test_other", "test_1h"))
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
        split=args.dataset_split,
        num_proc=4,
    )
    mt_model = Text2Text(
        mt_train_config="exp/wavlm_baseline/config.yaml",
        mt_model_file="exp/wavlm_baseline/valid.acc.ave_10best.pth",
        beam_size=args.beam_size,
        ctc_weight=0.0,
        lm_weight=0.0,
        device=device
    )

    unit_model = instantiate(config.model)
    if args.ckpt:
        d = torch.load(args.ckpt)
        if 'layer_norm..mean' in d:
            d['layer_norm.mean'] = d.pop('layer_norm..mean')
            d['layer_norm.var'] = d.pop('layer_norm..var')
        unit_model.load_state_dict(d)
    unit_model.to(device)

    tokenizer = build_tokenizer(
        token_type="bpe",
        bpemodel="ondevice_demo/baseline/data/token_list/src_bpe_unigram3000_rm_wavlm_large_21_km2000/bpe.model",
    )
    converter = TokenIDConverter(token_list="ondevice_demo/baseline/data/token_list/src_bpe_unigram3000_rm_wavlm_large_21_km2000/tokens.txt")

    # Steo 2. Setup directories
    # eval_dir = f"{args.expdir}/eval_results/{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    eval_dir = f"{args.output_dir}/eval_results/{args.dataset_split}/{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
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
        audio = torch.from_numpy(data['audio']).to(device)

        # Forward pass
        with torch.no_grad():
            units = unit_model.inference(audio[None]).cpu().detach().numpy()

        deduplicated_units = [x[0] for x in groupby(units)]
        cjk_units = "".join([chr(int("4e00", 16) + c) for c in deduplicated_units])
        bpe_tokens = tokenizer.text2tokens(cjk_units)
        bpe_tokens = converter.tokens2ids(bpe_tokens)
        bpe_tokens = torch.LongTensor(bpe_tokens).to("cuda")

        results = mt_model(bpe_tokens)

        ids.append(data['id'])
        gts.append(data['text'])
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
