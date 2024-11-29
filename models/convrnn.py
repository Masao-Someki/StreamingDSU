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

from models.ASRBaseline import ApplyKmeans
from models.ctc import CTC


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super(GLU, self).__init__()
        self.out_dim = out_dim

        self.conv = nn.Conv1d(in_dim, out_dim * 2, kernel_size, dilation=dilation)

    def forward(self, x):
        x = self.conv(x) # (B, D, L) to (B, D*2, L)
        x_base = x[:, :self.out_dim]
        x_sigma = x[:, self.out_dim:]

        return x_base * torch.sigmoid(x_sigma)


class DilConv(nn.Module):
    def __init__(self, in_dim=80, kernel_size=3, n_convs=2, conv_type='glu'):
        super(DilConv, self).__init__()

        # DilConv
        self.convs = nn.ModuleList()

        for i in range(n_convs):
            if conv_type == 'glu':
                # Gated Linear Units
                self.convs += [
                        GLU(in_dim * kernel_size ** i,
                            in_dim * kernel_size ** (i + 1),
                            kernel_size,
                            kernel_size ** i
                        )
                ]
            elif conv_type == 'linear':
                # Normal conv1d
                self.convs += [
                        nn.Conv1d(
                            in_dim * kernel_size ** i,
                            in_dim * kernel_size ** (i + 1),
                            kernel_size,
                            dilation=kernel_size ** i
                        )
                ]
            else:
                raise ValueError('conv type %s is not supported now.' % conv_type)

        # dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for i, layer in enumerate(self.convs):
            x = layer(x)
        x = self.dropout(x)
        return x


class RNN(nn.Module):
    def __init__(self, in_dim, kernel_size, n_convs,
        rnn_type="lstm", h_units=512, n_layers=1, output_hidden_state=False):
        super(RNN, self).__init__()
        self.in_dim = in_dim * kernel_size ** n_convs
        self.n_layers = n_layers
        self.output_hidden_state = output_hidden_state

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                    self.in_dim,
                    hidden_size=h_units,
                    num_layers=n_layers,
                    bidirectional=False,
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                    self.in_dim,
                    hidden_size=h_units,
                    num_layers=n_layers,
                    bidirectional=False,
            )

        self.dropout = nn.Dropout(p=0.5)
        self.h_size = (n_layers * 1, 1, h_units)

    def forward(self, x, h=None, c=None):
        # x: (L, B, D)
        if h is None:
            ret,_ = self.rnn(x)
            ret = self.dropout(ret)
            h = None
        else:
            ret, (h, c) = self.rnn(x, (h, c))
        
        if self.output_hidden_state:
            return ret, (h, c)
        else:
            return ret


class Exp4(nn.Module):
    def __init__(
        self,
        in_dim=128, kernel_size=3, n_convs=2, conv_type='glu',
        vocab_size=2000, rnn_type="lstm", h_units=512, n_layers=1,
        output_hidden_state=False,
        **kwargs,
    ):
        super().__init__()
        model = S3prlFrontend(
            fs=16000,
            frontend_conf={
                "upstream": "wavlm_large",
            },
            download_dir="./hub",
            multilayer_feature=False,
            layer=21,
        )
        self.feature_extractor = model.upstream.upstream.model.feature_extractor
        self.feat_normalizer = model.upstream.upstream.model.layer_norm
        self.feature_extractor.eval()
        self.feat_normalizer.eval()

        self.linear = Conv1d1x1(512, in_dim, bias=True)
        self.conv = DilConv(
            in_dim, kernel_size, n_convs, conv_type
        )

        rec_field = kernel_size ** n_convs
        print(f"Receptive field: {rec_field * 50} ms")
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
        x = self.rnn(x.transpose(0, 1)).transpose(0, 1) # (L, B, D)
        x = self.out_linear(x) # (B, L, vocab_size)
        ce_loss = self.loss(x.transpose(1, 2), text[:, :x.size(1)])

        acc = 0
        for b in range(x.shape[0]):
            text_length = torch.sum(text[b][:x.size(1)] != -1)
            selected_cls = torch.argmax(x[b][:x.size(1)], dim=-1)
            acc += torch.sum(selected_cls == text[b][:x.size(1)]) / text_length
        acc /= x.shape[0]
        
        return ce_loss, {"loss": ce_loss.item(), "acc": acc}


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


class Exp4CTC(nn.Module):
    def __init__(
        self,
        in_dim=128, kernel_size=3, n_convs=2, conv_type='glu',
        vocab_size=2000, rnn_type="lstm", h_units=512, n_layers=1,
        **kwargs,
    ):
        super().__init__()
        model = S3prlFrontend(
            fs=16000,
            frontend_conf={
                "upstream": "wavlm_large",
            },
            download_dir="./hub",
            multilayer_feature=False,
            layer=21,
        )
        self.feature_extractor = model.upstream.upstream.model.feature_extractor
        self.feat_normalizer = model.upstream.upstream.model.layer_norm
        self.feature_extractor.eval()
        self.feat_normalizer.eval()

        self.linear = Conv1d1x1(512, in_dim, bias=True)
        self.conv = DilConv(
            in_dim, kernel_size, n_convs, conv_type
        )

        rec_field = kernel_size ** n_convs
        print(f"Receptive field: {rec_field * 50} ms")
        self.padding = nn.ConstantPad1d(int((rec_field - 1) / 2), 0)
        self.rnn = RNN(in_dim, kernel_size, n_convs,
            rnn_type, h_units, n_layers)

        self.out_linear = nn.Linear(h_units, vocab_size)
        self.ctc = CTC(h_units, vocab_size)

        # loss
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.ctc_loss = nn.CTCLoss()

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
            # padded_wav = pad_sequence(wavs, batch_first=True)
            feats = []
            feat_lengths = []
            for wav in wavs:
                feat = self.feature_extractor(wav[None]) # (B, T, D)
                feat = self.feat_normalizer(feat.transpose(1, 2)).transpose(1, 2)
                feats.append(feat[0])
                feat_lengths.append(feat.shape[2])
            
            feats = pad_list(feats, pad_value=0).to(speech.device)
            feat_lengths = torch.LongTensor(feat_lengths).to(speech.device)

        feats = self.linear(feats)
        padded = self.padding(feats) # (B, D, T+padding)
        x = self.conv(padded).transpose(1, 2) # (B, L, D)
        x = self.rnn(x.transpose(0, 1)).transpose(0, 1) # (L, B, D)
        units = self.out_linear(x) # (B, L, vocab_size)
        ce_loss = self.loss(units.transpose(1, 2), text[:, :x.size(1)])

        # ctc loss
        dedup_label = []
        dedup_lens = []
        for b in range(text.shape[0]):
            dedup_l = torch.LongTensor([t[0] for t in groupby(text[b][:x.size(1)])])
            dedup_label.append(dedup_l)
            dedup_lens.append(len(dedup_l))

        dedup_label = pad_list(dedup_label, -1).to(units.device)
        dedup_lens = torch.LongTensor(dedup_lens).to(units.device)

        ctc_output = self.ctc(x)
        ctc_loss = self.ctc_loss(
            ctc_output.transpose(0, 1),
            dedup_label,
            feat_lengths,
            dedup_lens, # for removing the last token
        )

        acc = 0
        for b in range(x.shape[0]):
            text_length = torch.sum(text[b][:x.size(1)] != -1)
            selected_cls = torch.argmax(x[b][:x.size(1)], dim=-1)
            acc += torch.sum(selected_cls == text[b][:x.size(1)]) / text_length
        acc /= x.shape[0]
        
        return ce_loss + ctc_loss, {
            "unit_loss": ce_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "acc": acc
        }


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
