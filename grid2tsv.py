import argparse
import os
from glob import glob
from tqdm import tqdm

import textgrid
from datasets import load_dataset


def parse_args():
    # get directory path.
    parser = argparse.ArgumentParser(description="Train a GAN-TTS model")
    parser.add_argument("--split", type=str, default="./exp", help="Path to save the textgrid information")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print("Experiment directory:", args.split)

    # get speakers
    # label dataset

    ds = load_dataset(
        "juice500/DSUChallenge2024-wavlm_large-l21-km2000",
        split=f"{args.split}"
    ).sort('id')
    id2phones = {}
    id2units = {}

    for data in tqdm(ds):
        units = data["units"]
        aid = data["id"]
        spk = aid.split("-")[0]
        chapter = aid.split("-")[1]
        utt_id = aid.split("-")[2]

        tg = textgrid.TextGrid.fromFile(f'{args.split}/{spk}/{chapter}/{spk}-{chapter}-{utt_id}.TextGrid')
        phones = ["" for _ in range(len(units))]

        for interval in tg[1].intervals:
            xmin = float(interval.minTime)
            xmax = float(interval.maxTime)
            start_idx = int(xmin * 1000 // 20)
            end_idx = int(xmax * 1000 // 20)
            if interval.mark == "" or interval.mark.islower():
                mark = "sil"
            else:
                mark = interval.mark

            phones[start_idx:end_idx] = [mark] * (end_idx - start_idx)
        
        id2phones[f"{spk}-{chapter}-{utt_id}"] = phones
        id2units[f"{spk}-{chapter}-{utt_id}"] = units
    
    with open(f"{args.split}_phoneme.txt", "w") as f:
        for aid, phones in id2phones.items():
            f.write(f"{aid}\t{' '.join(phones)}\n")
    
    with open(f"{args.split}_units.txt", "w") as f:
        for aid, units in id2units.items():
            f.write(f"{aid}\t{' '.join(map(str, units))}\n")
        
