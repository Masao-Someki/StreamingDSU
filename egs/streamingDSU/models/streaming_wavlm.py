import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.train.abs_espnet_model import AbsESPnetModel
from src.wavlm import WavLM, WavLMConfig


class StreamingWavLM(AbsESPnetModel):
    def __init__(
        self,
        ckpt_path: str,
        vocab_size=2000,
        **kwargs,
    ):
        super().__init__()
        checkpoint = torch.load(ckpt_path)
        self.cfg = WavLMConfig(checkpoint['cfg'])
        self.cfg.extractor_mode = "no_noramalize"
        self.model = WavLM(self.cfg)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.layer_weight = nn.Parameter(torch.ones(21), requires_grad=True)
        self.projector = nn.Linear(1024, vocab_size)

        # loss
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
        if self.cfg.normalize:
            speech = torch.nn.functional.layer_norm(speech , speech.shape)

        rep, layer_results = self.model.extract_features(
            speech,
            output_layer=[i for i in range(21)],
            ret_layer_results=True
        )[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        norm_output_weights = F.softmax(self.layer_weight, dim=0)
        weighted_output = [
            output * weight
            for output, weight in zip(layer_reps, norm_output_weights)
        ]
        weighted_output = torch.stack(weighted_output).sum(dim=0)
        x = self.projector(weighted_output) # (B, T, vocab_size)

        ce_loss = self.loss(x.transpose(1, 2), text[:, :x.size(1)])

        acc = 0
        for b in range(x.shape[0]):
            text_length = torch.sum(text[b][:x.size(1)] != -1)
            selected_cls = torch.argmax(x[b][:x.size(1)], dim=-1)
            acc += torch.sum(selected_cls == text[b][:x.size(1)]) / text_length
        acc /= x.shape[0]
        
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
