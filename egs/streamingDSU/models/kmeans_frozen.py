from itertools import groupby
import logging

import joblib
import sentencepiece as spm
import torch
import torch.nn as nn
import librosa
import joblib
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.bin.mt_inference import Text2Text
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.abs_espnet_model import AbsESPnetModel


class FrozenSSLWithKmeans(AbsESPnetModel):
    def __init__(
        self,
        huggingface_model: str,
        kmeans_model: str,
        layer: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.model = S3prlFrontend(
            fs=16000,
            frontend_conf={
                "upstream": huggingface_model,
            },
            download_dir="./hub",
            multilayer_feature=False,
            layer=layer,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        km_model = joblib.load(kmeans_model)
        C = km_model.cluster_centers_.transpose()
        Cnorm = (C**2).sum(0, keepdims=True)

        self.C = nn.Parameter(torch.from_numpy(C), requires_grad=False)
        self.Cnorm = nn.Parameter(torch.from_numpy(Cnorm), requires_grad=False)


    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """
        speech: (B, T_s)
        text: (B, T_u)
        """
        assert NotImplementedError

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}

    def inference(
        self,
        speech: torch.Tensor,
        **kwargs,
    ):
        assert len(speech) == 1
        speech_lengths = torch.LongTensor([len(speech[0])]).to(speech.device)
        with torch.no_grad():
            feats, feats_lens = self.model(speech, speech_lengths) # (B, T, D)
        feats = feats[0]
        dist = feats.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(feats, self.C) + self.Cnorm
        return dist.argmin(dim=1)
