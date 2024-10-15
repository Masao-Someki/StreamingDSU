import numpy as np
import torch
import datasets
from datasets import Audio


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, num_proc=1):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        self.units_data = datasets.load_dataset(
            "juice500/DSUChallenge2024-wavlm_large-l21-km2000"
        )[split].sort('id')
        self.audio_data = datasets.load_dataset(
            "espnet/DSUChallenge2024"
        )[split].cast_column("audio", Audio(decode=False)).sort('id')

        # Check ids and align if not
        units_ids = set(self.units_data['id'])
        audio_ids = set(self.audio_data['id'])
        if not units_ids == audio_ids:
            print("Warning: units and audio ids do not match. Aligning...")
            ids = units_ids.intersection(audio_ids)
            self.units_data = self.units_data.filter(
                lambda example: example["id"] in ids,
                num_proc=num_proc
            )
            self.audio_data = self.audio_data.filter(
                lambda example: example["id"] in ids,
                num_proc=num_proc
            )
        
        self.audio_data = self.audio_data.cast_column("audio", Audio(decode=True))
        
    def __len__(self):
        return len(self.units_data)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        units = self.units_data[idx]
        assert audio["id"] == units["id"], "IDs do not match"

        return {
            "audio": audio["audio"]['array'].astype(np.float32),
            "units": units["units"],
        }
