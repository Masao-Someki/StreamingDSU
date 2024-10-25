import numpy as np
import torch
import datasets
from datasets import Audio

from espnet2.train.preprocessor import CommonPreprocessor


token_list = [
    "<blank>",
"<unk>",
"' '",
"AH0",
"N",
"T",
"D",
"S",
"R",
"L",
"DH",
"K",
"Z",
"IH1",
"IH0",
"M",
"EH1",
"W",
"P",
"AE1",
"AH1",
"V",
"ER0",
"F",
"','",
"AA1",
"B",
"HH",
"IY1",
"UW1",
"IY0",
"AO1",
"EY1",
"AY1",
".",
"OW1",
"SH",
"NG",
"G",
"ER1",
"CH",
"JH",
"Y",
"AW1",
"TH",
"UH1",
"EH2",
"OW0",
"EY2",
"AO0",
"IH2",
"AE2",
"AY2",
"AA2",
"UW0",
"EH0",
"OY1",
"EY0",
"AO2",
"ZH",
"OW2",
"AE0",
"UW2",
"AH2",
"AY0",
"IY2",
"AW2",
"AA0",
"''''",
"ER2",
"UH2",
"'?'",
"OY2",
"'!'",
"AW0",
"UH0",
"OY0",
"..",
"<sos/eos>",
]


class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, num_proc=1):
        assert split is not None, "Split must be provided"
        self.split = split

        # For ASR we use the following two datasets:
        self.data = datasets.load_dataset("ms180/ljspeech_with_dsu")[split]

        self.preprocessor = CommonPreprocessor(
            train=False,
            token_type="phn",
            token_list=token_list,
            bpemodel=None,
            non_linguistic_symbols=None,
            text_cleaner="tacotron",
            g2p_type="g2p_en",
        )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        return {
            "audio": data["audio"]['array'].astype(np.float32),
            "text": self.preprocessor("<dummy>", dict(text=data['text']))["text"],
            "units": np.array(data["units"].split()).astype(np.int64),
        }
