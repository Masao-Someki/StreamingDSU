import torch
import torch.nn as nn
from espnet2.train.abs_espnet_model import AbsESPnetModel


class TemplateModel(AbsESPnetModel):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.encoder.dim, config.decoder.dim),
            nn.ReLU(),
            nn.Linear(config.decoder.dim, config.decoder.dim),
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        # Your implementation here
        return 0.0, {}

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        # Your implementation here
        return 0.0, {}
