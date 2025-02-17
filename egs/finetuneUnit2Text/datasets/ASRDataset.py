from collections import defaultdict
import functools
import random
from itertools import groupby
import numpy as np
import torch
import datasets
from datasets import Audio
from tqdm import tqdm
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self,
        split=None,
        num_proc=1, 
        src_bpe_path: str = None,
        src_token_list_path: str = None,
        tgt_bpe_path: str = None,
        tgt_token_list_path: str = None,
        unit_path: str = None,  # Path to the unit speech dataset
    ):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        with open(unit_path, "r") as f:
            self.units_data = {}
            for row in f.readlines():
                key, units = row.split(maxsplit=1)
                self.units_data[key] = units.strip()

        keys = set(self.units_data.keys())
        self.audio_data = datasets.load_dataset(
            "espnet/DSUChallenge2024"
        )[split].remove_columns(["audio"]).filter(
            lambda example: example["id"] in keys,
            num_proc=num_proc,
        )

        self.tokenizers = {
            "src": build_tokenizer(
                token_type="bpe",
                bpemodel=src_bpe_path,
            ),
            "tgt": build_tokenizer(
                token_type="bpe",
                bpemodel=tgt_bpe_path,
            ),
        }
        self.converters = {
            "src": TokenIDConverter(token_list=src_token_list_path),
            "tgt": TokenIDConverter(token_list=tgt_token_list_path),
        }
        self.tokenizers["src"]._build_sentence_piece_processor()
        self.tokenizers["tgt"]._build_sentence_piece_processor()

    def __len__(self):
        return len(self.audio_data)

    # @functools.lru_cache(maxsize=32, typed=False)
    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        units = self.units_data[audio["id"]]

        unit_text = "".join([x[0] for x in groupby(units)])
        unit_str = self.tokenizers["src"].text2tokens(unit_text)
        unit_bpe = self.converters["src"].tokens2ids(unit_str)

        text = audio["text"]
        text_token = self.tokenizers["tgt"].text2tokens(text)
        text_bpe = self.converters["tgt"].tokens2ids(text_token)

        return {
            "id": audio["id"],
            "units": np.array(unit_bpe).astype(np.int64),
            "text": np.array(text_bpe).astype(np.int64),
        }
