from itertools import groupby
import logging

import joblib
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import librosa
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.gan_tts.wavenet.residual_block import Conv1d1x1
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.global_mvn import GlobalMVN

from .utils import ApplyKmeans, CentroidLoss
from .convrnn import DilConv, RNN


class ContinuousLoss(nn.Module):
    def __init__(self):
        super(ContinuousLoss, self).__init__()
    
    def forward(self, y_pred, y_true, y_lengths):
        # Compute continuous loss
        # y_pred: (B, T, D)
        # y_true: (B, T, D)
        # y_lengths: (B,)
        continuous_loss = F.l1_loss(y_pred, y_true, reduction='none')
        mask = torch.arange(y_pred.size(1), device=y_lengths.device).unsqueeze(0) >= y_lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        continuous_loss = continuous_loss.masked_fill(mask, 0.0)
        continuous_loss = continuous_loss.sum(dim=2).mean()
        return continuous_loss


class ConvRNNMultiLayer(AbsESPnetModel):
    def __init__(
        self,
        in_dim=128, kernel_size=3, n_convs=2, conv_type='glu',
        vocab_size=2000, rnn_type="lstm", h_units=512, n_layers=1,
        output_hidden_state=False, predict_layers=[21],
        **kwargs,
    ):
        super().__init__()
        if len(predict_layers) == 1:
            model = S3prlFrontend(
                fs=16000,
                frontend_conf={
                    "upstream": "wavlm_large",
                },
                download_dir="./hub",
                multilayer_feature=False,
                layer=predict_layers[0],
            )
        else:
            model = S3prlFrontend(
                fs=16000,
                frontend_conf={
                    "upstream": "wavlm_large",
                },
                download_dir="./hub",
                multilayer_feature=True,
                layers=predict_layers,
            )
        self.feature_extractor = model.upstream.upstream.model.feature_extractor
        self.feat_normalizer = model.upstream.upstream.model.layer_norm
        self.feature_extractor.eval()
        self.feat_normalizer.eval()
        self.model.eval()

        self.linear = Conv1d1x1(512, in_dim, bias=True)
        self.conv = DilConv(
            in_dim, kernel_size, n_convs, conv_type
        )

        rec_field = kernel_size ** n_convs
        print(f"Receptive field: {rec_field * 20} ms")
        self.padding = nn.ConstantPad1d(int((rec_field - 1) / 2), 0)
        self.rnn = RNN(in_dim, kernel_size, n_convs,
            rnn_type, h_units, n_layers, output_hidden_state)

        self.predict_layers = {}
        for layer_idx in predict_layers:
            self.predict_layers[layer_idx] = nn.Linear(h_units, 1024)

        # loss
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.continuous_loss = ContinuousLoss()

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
            wavs = [F.layer_norm(wav, wav.shape) for wav in speech]
            wav_padding_mask = ~torch.lt(
                torch.arange(max(speech_lengths)).unsqueeze(0).to(speech.device),
                speech_lengths.unsqueeze(1),
            )
            padded_wav = pad_sequence(wavs, batch_first=True)
            feats = self.feature_extractor(padded_wav) # (B, T, D)
            feats = self.feat_normalizer(feats.transpose(1, 2)).transpose(1, 2)

        feats = self.linear(feats)
        padded = self.padding(feats) # (B, D, T+padding)
        x = self.conv(padded).transpose(1, 2) # (B, L, D)
        x = self.rnn(x.transpose(0, 1)).transpose(0, 1) # (L, B, D) -> (B, L ,D)
        units = self.out_linear(x) # (B, L, vocab_size)
        ce_loss = self.loss(units.transpose(1, 2), text[:, :units.size(1)])

        intermediate_loss = 0
        intermediate_feats = self.model(speech, speech_lengths)
        for layer_idx, layer in self.predict_layers.items():
            predicted_feats = layer(x)
            intermediate_loss += self.continuous_loss(
                predicted_feats,
                intermediate_feats,
                text_lengths,
            )
        
        intermediate_loss /= len(self.predict_layers)

        acc = 0
        for b in range(units.shape[0]):
            text_length = torch.sum(text[b][:units.size(1)] != -1)
            selected_cls = torch.argmax(units[b][:units.size(1)], dim=-1)
            acc += torch.sum(selected_cls == text[b][:units.size(1)]) / text_length
        acc /= units.shape[0]
        
        return ce_loss + intermediate_loss * 5, {"ce_loss": ce_loss.item(), "intermediate_loss": intermediate_loss.item(), "acc": acc}, None

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
