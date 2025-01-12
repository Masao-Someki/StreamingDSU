import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.asr.frontend.default import DefaultFrontend

from egs.streamingDSU.models.convrnn import RNN

# Generator


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        
        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels//2, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class SoundStreamRNNEncoder(AbsESPnetModel):
    def __init__(self, in_channels=80, C=128, out_channels=2000, rnn_type='gru',
        h_units=2048, n_layers=1, output_hidden_state=False):
        super().__init__()
        self.frontend = DefaultFrontend(
            n_fft=400,
            hop_length=320,
            center=False,
            n_mels=in_channels,
        )

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2*C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=4*C, stride=4),
            nn.ELU(),
            EncoderBlock(out_channels=8*C, stride=5),
            nn.ELU(),
            EncoderBlock(out_channels=16*C, stride=8),
            nn.ELU(),
            CausalConv1d(in_channels=16*C, out_channels=out_channels, kernel_size=3)
        )
        self.out_linear = nn.Linear(out_channels, out_channels)
        self.rnn = RNN(out_channels, 1, 1,
            rnn_type, h_units, n_layers, output_hidden_state)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        feats, feat_lengths = self.frontend(speech, speech_lengths) # (B, T, D)
        x = self.layers(feats.transpose(1, 2)).transpose(1, 2) # (B, L, D)
        x = self.out_linear(x) # (B, L, vocab_size)
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
