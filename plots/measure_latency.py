import time
import numpy as np
from transformers import AutoModel
import torch
from tqdm import tqdm
import pickle
import pandas as pd


if __name__ == "__main__":
    model = AutoModel.from_pretrained("microsoft/wavlm-large")
    layers = [21, 18, 15, 12]
    frames = np.array([64, 32, 16, 8, 4, 2, 1, 0]) * 2 + 1

    data = []
    for layer in layers:
        del model.encoder.layers[layer:]
        for frame in frames:
            for _ in tqdm(range(100)):
                model(torch.rand(1, 80 + 320 * frame))

            start = time.time()
            for _ in tqdm(range(100)):
                model(torch.rand(1, 80 + 320 * frame))
            dur = (time.time() - start) / 100
            data.append({
                "layer": layer,
                "frame": frame,
                "dur": dur,
            })

    pd.DataFrame(data).to_csv("latency.csv", index=False)
