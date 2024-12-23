import argparse
from argparse import Namespace
from pprint import pprint
from datetime import datetime
import os
from glob import glob
import joblib

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from egs.streamingDSU import AudioDataset
from espnet2.asr.frontend.s3prl import S3prlFrontend
import espnetez as ez

import logging

logging.basicConfig(level=logging.INFO)


# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network for speech synthesis')

    # general arguments
    parser.add_argument('--savedir', type=str, required=True, help='Path to the save kmeans models')
    parser.add_argument('--layer', type=int, default=21, help='Index of the encoder layer to use for k-means clustering')
    parser.add_argument('--n_clusters', type=int, default=2000, help='Number of clusters for k-means clustering')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Step 0. Parse command-line arguments and load configuration
    args = parse_args()
    train_dataset = AudioDataset(split="train")
    
    print(f'Number of training samples: {len(train_dataset)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2. Setup model
    minibatch_size = 10000
    model = S3prlFrontend(
        fs=16000,
        frontend_conf={
            "upstream": "wavlm_large",
        },
        download_dir="./hub",
        multilayer_feature=False,
        layer=args.layer,
    )
    model.to(device)

    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        init="k-means++",
        batch_size=minibatch_size,
        verbose=1,
        compute_labels=False,
        random_state=2024,
    )

    minibatch = torch.zeros(minibatch_size, 1024, requires_grad=False).to(device)
    mb_idx = 0

    for epoch in range(3): # train 3 epochs
        for n_iter, b in enumerate(train_dataset):
            # Step 3. Forward pass
            speech = torch.from_numpy(b["speech"])[None].to(device)
            speech_lengths = torch.LongTensor([len(b["speech"])]).to(device)
            
            with torch.no_grad():
                feats, feats_lengths = model(speech, speech_lengths)

            for i, feats_len in enumerate(feats_lengths):
                if minibatch_size - mb_idx >= feats_len:
                    minibatch[mb_idx:mb_idx+feats_len] = feats[i, :feats_len]
                    mb_idx += feats_len
                    if mb_idx == minibatch_size:
                        km.partial_fit(minibatch.cpu().numpy())
                        mb_idx = 0
                else:
                    used_feats_len = minibatch_size - mb_idx
                    minibatch[mb_idx:] = feats[i, :used_feats_len]
                    km.partial_fit(minibatch.cpu().numpy())
                    mb_idx = feats_len - used_feats_len
                    minibatch[:mb_idx] = feats[i, used_feats_len:feats_len]

            if n_iter % 1000 == 0:
                if len(glob(f"{args.savedir}/kmeans.{args.n_clusters}.tmp.*.pkl")) > 0:
                    os.remove(glob(f"{args.savedir}/kmeans.{args.n_clusters}.tmp.*.pkl")[0])
                joblib.dump(km, f"{args.savedir}/kmeans.{args.n_clusters}.tmp.{epoch}_{n_iter}.pkl")
    
    if mb_idx > 0:
        km.partial_fit(minibatch[:mb_idx].cpu().numpy())
    
    joblib.dump(km, f"{args.savedir}/kmeans.{args.n_clusters}.mdl")
