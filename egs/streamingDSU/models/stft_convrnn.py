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

from .utils import ApplyKmeans
from .convrnn import DilConv, RNN


class STFTConvRNN(AbsESPnetModel):
    def __init__(
        self,
        in_dim=80, kernel_size=3, n_convs=2, conv_type='glu',
        vocab_size=2000, rnn_type="lstm", h_units=512, n_layers=1,
        output_hidden_state=False,
        **kwargs,
    ):
        super().__init__()
        self.frontend = DefaultFrontend(
            n_fft=512,
            hop_length=320,
            center=False,
            n_mels=in_dim,
        )
        self.normalize = GlobalMVN(stats_file="feats_stats.npz")

        self.conv = DilConv(
            in_dim, kernel_size, n_convs, conv_type
        )

        rec_field = kernel_size ** n_convs
        print(f"Receptive field: {rec_field * 20} ms")
        self.padding = nn.ConstantPad1d(int((rec_field - 1) / 2), 0)
        self.rnn = RNN(in_dim, kernel_size, n_convs,
            rnn_type, h_units, n_layers, output_hidden_state)

        self.out_linear = nn.Linear(h_units, vocab_size)

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
        with torch.no_grad():
            feats, feat_lengths = self.frontend(speech, speech_lengths) # (B, T, D)
            feats, feat_lengths = self.normalize(feats, feat_lengths) # (B, T, D)

        padded = self.padding(feats.transpose(1, 2)) # (B, D, T+padding)
        x = self.conv(padded).transpose(1, 2) # (B, L, D)
        x = self.rnn(x.transpose(0, 1)).transpose(0, 1) # (L, B, D)
        x = self.out_linear(x) # (B, L, vocab_size)

        ce_loss = self.loss(x.transpose(1, 2), text[:, :x.size(1)])

        acc = 0
        for b in range(x.shape[0]):
            text_length = torch.sum(text[b][:x.size(1)] != -1)
            selected_cls = torch.argmax(x[b][:x.size(1)], dim=-1)
            acc += torch.sum(selected_cls == text[b][:x.size(1)]) / text_length
        acc /= x.shape[0]
        
        return ce_loss, {"loss": ce_loss.item(), "acc": acc}, None


    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        with torch.no_grad():
            feats = self.feature_extractor(speech) # (B, T, D)

        padded = self.padding(feats.transpose(1, 2)) # (B, D, T+padding)
        x = self.conv(padded).transpose(1, 2) # (B, L, D)
        x = self.rnn(x.transpose(0, 1)).transpose(0, 1) # (L, B, D)
        x = self.out_linear(x) # (B, L, vocab_size)
        units = torch.argmax(x[0], dim=-1) # We don't have to calculate softmax.

        # De-duplicate units
        deduplicated_units = [x[0] for x in groupby(units)]

        # units to cjk characters and apply BPE
        cjk_units = "".join([chr(int("4e00", 16) + c) for c in units])
        cjk_tokens_hyp = "".join([chr(int("4e00", 16) + c) for c in deduplicated_units])
        bpe_tokens = self.tokenizer.text2tokens(cjk_tokens_hyp)
        bpe_tokens = self.converter.tokens2ids(bpe_tokens)
        bpe_tokens = torch.Tensor(bpe_tokens).to(speech.device)

        # Inference using the MT model
        hyp_results = self.mt_model(bpe_tokens)

        # Inference with correct units
        deduplicated_units = [x[0] for x in groupby(texts[0])]
        cjk_tokens = "".join([chr(int("4e00", 16) + c) for c in deduplicated_units])
        bpe_tokens = self.tokenizer.text2tokens(cjk_tokens)
        bpe_tokens = self.converter.tokens2ids(bpe_tokens)
        bpe_tokens = torch.Tensor(bpe_tokens).to(speech.device)
        results = self.mt_model(bpe_tokens)


        return {
            "text": hyp_results[0][0],
            "true_label": results[0][0],
            "units": cjk_units,
            "deduplicated_units": cjk_tokens_hyp,
        }

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}
