import numpy as np
import torch
import datasets


class SVSDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, num_proc=1):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        self.data = datasets.load_dataset(
            "jhansss/opencpop_dsu", cache_dir="cache", split=split
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "audio": data["audio"]["array"].astype(np.float32),
            "units": np.array(data["token_wavlm_large_1024_6"].split()).astype(
                np.int64
            ),
        }
