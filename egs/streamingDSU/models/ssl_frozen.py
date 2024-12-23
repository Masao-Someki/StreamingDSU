from itertools import groupby
import logging

import joblib
import sentencepiece as spm
import torch
import torch.nn as nn
import librosa
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.bin.mt_inference import Text2Text
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.abs_espnet_model import AbsESPnetModel



class FrozenSSLWithLinear(AbsESPnetModel):
    def __init__(
        self,
        huggingface_model: str,
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

        # Frozen parameter
        self.clustering_head = nn.Linear(1024, 2000)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)


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
        with torch.no_grad():
            feats, feats_lens = self.model(speech, speech_lengths) # (B, T, D)

        units = self.clustering_head(feats) # (B, T, Cluster)
        ce_loss = self.loss(units.transpose(1, 2), text)
        
        acc = 0
        for b in range(units.shape[0]):
            text_length = torch.sum(text[b] != -1)
            selected_cls = torch.argmax(units[b], dim=-1)
            acc += torch.sum(selected_cls == text[b]) / text_length
        
        acc /= units.shape[0]

        return ce_loss, {"loss": ce_loss.item(), "acc": acc}, None

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
