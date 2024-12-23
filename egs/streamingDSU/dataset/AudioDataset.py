import numpy as np
import torch
import datasets
from datasets import Audio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, num_proc=1):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        self.audio_data = datasets.load_dataset(
            "espnet/DSUChallenge2024"
        )[split].cast_column("audio", Audio(decode=False)).sort('id')

        self.audio_data = self.audio_data.cast_column("audio", Audio(decode=True))
        
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        return {
            "speech": audio["audio"]['array'].astype(np.float32),
        }
