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

from models.ASRBaseline import ApplyKmeans



class Exp2(nn.Module):
    def __init__(
        self,
        huggingface_model: str,
        checkpoint_path: str = None,
        config_path: str = None,
        cluster_checkpoint: str = None,
        bpemodel_path: str = None,
        token_list: str = None,
        kmeans_path: str = None,
        layer: int = 20,
        frozen_ssl: bool = False,
        use_centroid: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.huggingface_model = huggingface_model
        self.model = S3prlFrontend(
            fs=16000,
            frontend_conf={
                "upstream": huggingface_model,
            },
            download_dir="./hub",
            multilayer_feature=False,
            layer=layer,
        )
        # Frozen parameter
        self.frozen_ssl = frozen_ssl
        if frozen_ssl:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.clustering_head = nn.Linear(768, 2000)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.use_centroid = use_centroid

        if self.use_centroid:
            self.centroid_head = nn.Linear(768, 1024)
            self.centroid_loss = nn.L1Loss(768)

        if cluster_checkpoint is not None:
            d = torch.load(cluster_checkpoint)
            new_d = {}
            for k in d.keys():
                new_d[k.replace("model.", "")] = d[k]
            self.clustering_head.load_state_dict(new_d)

        # for inference
        self.mt_model = Text2Text(
            mt_train_config=config_path,
            mt_model_file=checkpoint_path,
            beam_size=10,
            ctc_weight=0.3,
            lm_weight=0.0,
        )
        self.quantizer = ApplyKmeans(kmeans_path, use_gpu=torch.cuda.is_available())
        self.C = self.quantizer.C.transpose(0, 1)
        self.tokenizer = build_tokenizer(
            token_type="bpe",
            bpemodel=bpemodel_path,
        )
        self.converter = TokenIDConverter(token_list=token_list)

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

        centroid_loss = 0.
        if self.use_centroid:
            C_label = self.C[text].detach()
            cents = self.centroid_head(feats)
            centroid_loss = self.centroid_loss(cents, C_label)

        acc = 0
        for b in range(units.shape[0]):
            text_length = torch.sum(text[b] != -1)
            selected_cls = torch.argmax(units[b], dim=-1)
            acc += torch.sum(selected_cls == text[b]) / text_length
        
        acc /= units.shape[0]
        loss = ce_loss + centroid_loss

        if self.use_centroid:
            return loss, {
                "ce_loss": ce_loss.item(),
                "centroid_loss": centroid_loss.item(),
                "acc": acc
            }
        else:
            return loss, {
                "ce_loss": ce_loss.item(),
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
        feats, feats_lens = self.model(speech, speech_lengths)
        units = self.clustering_head(feats)[0]
        units = torch.argmax(units, dim=-1) # We don't have to calculate softmax.

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
    
    def state_dict(self):
        if self.frozen_ssl:
            return self.clustering_head.state_dict()
        else:
            return {
                "model": self.clustering_head.state_dict(),
                "ssl": self.model.state_dict(),
                "ssl_model": self.huggingface_model,
            }


class Exp2TTSModel(nn.Module):
    def __init__(
        self,
        huggingface_model: str,
        checkpoint_path: str = None,
        config_path: str = None,
        cluster_checkpoint: str = None,
        bpemodel_path: str = None,
        token_list: str = None,
        kmeans_path: str = None,
        layer: int = 20,
        frozen_ssl: bool = False,
        use_centroid: bool = False,
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
        # Frozen parameter
        if frozen_ssl:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.clustering_head = nn.Linear(1024, 2000)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

        if self.use_centroid:
            self.centroid_head = nn.Linear(1024, 1024)
            self.centroid_loss = nn.L1Loss(1024)

        if cluster_checkpoint is not None:
            d = torch.load(cluster_checkpoint)
            new_d = {}
            for k in d.keys():
                new_d[k.replace("model.", "")] = d[k]
            self.clustering_head.load_state_dict(new_d)

        # for inference
        self.mt_model = Text2Text(
            mt_train_config=config_path,
            mt_model_file=checkpoint_path,
            beam_size=10,
            ctc_weight=0.3,
            lm_weight=0.0,
        )
        self.quantizer = ApplyKmeans(kmeans_path, use_gpu=torch.cuda.is_available())
        self.C = self.quantizer.C.transpose(0, 1)
        self.tokenizer = build_tokenizer(
            token_type="bpe",
            bpemodel=bpemodel_path,
        )
        self.converter = TokenIDConverter(token_list=token_list)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: torch.Tensor,
        **kwargs,
    ):
        with torch.no_grad():
            feats, feats_lens = self.model(speech, speech_lengths) # (B, T, D)

        units = self.clustering_head(feats) # (B, T, Cluster)
        ce_loss = self.loss(units.transpose(1, 2), sids)

        centroid_loss = 0
        if self.use_centroid:
            C_label = self.C[sids].detach()
            cents = self.centroid_head(feats)
            centroid_loss = self.centroid_loss(cents, C_label)

        acc = 0
        for b in range(units.shape[0]):
            text_length = torch.sum(sids[b] != -1)
            selected_cls = torch.argmax(units[b], dim=-1)
            acc += torch.sum(selected_cls == sids[b]) / text_length
        
        acc /= units.shape[0]
        loss = ce_loss + centroid_loss

        return loss, {"ce_loss": ce_loss.item(), "centroid_loss": centroid_loss.item(),"acc": acc}


    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: torch.Tensor,
        sids_lengths: torch.Tensor,
        **kwargs,
    ):
        feats, feats_lens = self.model(speech, speech_lengths)
        units = self.clustering_head(feats)[0]
        units = torch.argmax(units, dim=-1) # We don't have to calculate softmax.

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

    def state_dict(self, *args, **kwargs):
        return self.clustering_head.state_dict(*args, **kwargs)
