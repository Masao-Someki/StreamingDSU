from collections import defaultdict
import functools
import random
import numpy as np
import torch
import datasets
from datasets import Audio
from tqdm import tqdm


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, num_proc=1):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        self.units_data = datasets.load_dataset(
            "juice500/DSUChallenge2024-wavlm_large-l21-km2000"
        )[split]
        self.audio_data = datasets.load_dataset(
            "espnet/DSUChallenge2024"
        )[split]

        self._intersection(num_proc)
        self.units_data = self.units_data.sort("id")
        self.audio_data = self.audio_data.sort("id")

    def _intersection(self, num_proc):
        # Check ids and align if not
        units_ids = set(self.units_data["id"])
        audio_ids = set(self.audio_data["id"])
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

        self.indices = np.arange(len(self.audio_data))

    def __len__(self):
        return len(self.indices)

    @functools.lru_cache(maxsize=32, typed=False)
    def __getitem__(self, idx):
        idx = int(self.indices[idx])
        audio = self.audio_data[idx]
        units = self.units_data[idx]
        assert audio["id"] == units["id"], "IDs do not match"

        return {
            "id": audio["id"],
            "audio": audio["audio"]["array"].astype(np.float32),
            "units": units["units"],
            "text": audio["text"],
        }

    def sample(self, n_sample, seed=42):
        mask = np.zeros(len(self.units_data), dtype=bool)
        mask[:n_sample] = True
        np.random.default_rng(seed=seed).shuffle(mask)

        self.indices = np.array(self.indices)[mask]


class SlicedASRDataset(ASRDataset):
    def __init__(self, num_units, split=None, num_proc=1, window_size=320, stride_size=320, sample=None):
        assert split is not None, "Split must be provided"
        assert (num_units % 2) == 1

        max_frames = (window_size - stride_size) + (stride_size * num_units)

        # For ASR we use the following two datasets:
        self.units_data = datasets.load_dataset(
            "juice500/DSUChallenge2024-wavlm_large-l21-km2000"
        )[split]
        self.audio_data = datasets.load_dataset(
            "espnet/DSUChallenge2024"
        )[split].filter(lambda x: (len(x["audio"]["array"]) >= max_frames))

        self._intersection(num_proc)
        self.units_data = self.units_data.sort("id")
        self.audio_data = self.audio_data.sort("id")

        self.split = split
        self.num_units = num_units
        self.window_size = window_size
        self.stride_size = stride_size
        self._prepare_units(sample)

    def _prepare_units(self, sample, seed=42):
        self.indices, self.unit_locs, self.audio_l_locs, self.audio_r_locs = [], [], [], []

        half_window = self.num_units // 2
        l = list(range(len(self.units_data)))
        if sample is not None:
            mask = np.zeros(len(l), dtype=bool)
            mask[:sample] = True
            np.random.default_rng(seed=seed).shuffle(mask)
            l = np.array(l)[mask]
            
        for idx in tqdm(l, desc="Unit-wise slicing"):
            units = self.units_data[int(idx)]["units"]

            for loc in range(half_window, len(units) - half_window):
                self.indices.append(idx)
                self.unit_locs.append(loc)
                self.audio_l_locs.append((loc - half_window) * self.stride_size)
                self.audio_r_locs.append(self.window_size + (self.stride_size * (loc + half_window)))
        self.indices = np.array(self.indices)
        self.unit_locs = np.array(self.unit_locs)
        self.audio_l_locs = np.array(self.audio_l_locs)
        self.audio_r_locs = np.array(self.audio_r_locs)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = super().__getitem__(int(self.indices[idx]))

        loc = self.unit_locs[idx]
        l_loc = self.audio_l_locs[idx]
        r_loc = self.audio_r_locs[idx]

        return {
            "audio": row["audio"][l_loc:r_loc],
            "units": row["units"][loc:loc+1],
        }

    def sample(self, n_sample, seed=42):
        mask = np.zeros(len(self.indices), dtype=bool)
        mask[:n_sample] = True
        np.random.default_rng(seed=seed).shuffle(mask)

        self.indices = np.array(self.indices)[mask]
        self.unit_locs = np.array(self.unit_locs)[mask]
        self.audio_l_locs = np.array(self.audio_l_locs)[mask]
        self.audio_r_locs = np.array(self.audio_r_locs)[mask]


class SlicedASRTestDataset(SlicedASRDataset):
    def _prepare_units(self, sample):
        super()._prepare_units(sample)
        self.audio_indices = defaultdict(list)
        for i, index in enumerate(self.indices):
            self.audio_indices[index].append(i)
        self.audio_keys = list(self.audio_indices.keys())

    def sample(self, n_sample, seed=42):
        l = list(self.audio_indices.keys())
        np.random.default_rng(seed=seed).shuffle(l)
        del l[n_sample:]

        self.audio_indices = {
            k: v for k, v in self.audio_indices.items()
            if k in l
        }
        self.audio_keys = list(self.audio_indices.keys())

    def __len__(self):
        return len(self.audio_keys)

    def __getitem__(self, idx):
        idx = self.audio_keys[idx]
        row = super(SlicedASRDataset, self).__getitem__(int(idx))

        return {
            "audio": row["audio"],
            "units": row["units"],
            "audio_l_locs": self.audio_l_locs[self.audio_indices[idx]],
            "audio_r_locs": self.audio_r_locs[self.audio_indices[idx]],
            "text": row["text"],
        }
