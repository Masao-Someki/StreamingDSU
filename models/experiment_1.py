from itertools import groupby

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



class Exp1(nn.Module):
    def __init__(
        self,
        huggingface_model: str,
        # checkpoint_path: str,
        # config_path: str,
        # kmeans_path: str,
        # bpemodel_path: str,
        # token_list: str,
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

        # for inference
        # self.mt_model = Text2Text(
        #     mt_train_config=config_path,
        #     mt_model_file=checkpoint_path,
        #     beam_size=10,
        #     ctc_weight=0.3,
        #     lm_weight=0.0,
        # )
        # self.quantizer = ApplyKmeans(kmeans_path, use_gpu=use_gpu_kmeans)
        # self.tokenizer = build_tokenizer(
        #     token_type="bpe",
        #     bpemodel=bpemodel_path,
        # )
        # self.converter = TokenIDConverter(token_list=token_list)

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

        return ce_loss, {"loss": ce_loss.item(), "acc": acc}


    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        if self.speed_up != 1.0:
            speech = librosa.effects.time_stretch(speech[0].cpu().numpy(), rate=2.0)
            speech = torch.from_numpy(speech).unsqueeze(0).to(speech_lengths.device)
            speech_lengths = torch.tensor([speech.shape[1]]).to(speech_lengths.device)
        feats, feats_lens = self.model(speech, speech_lengths)
        units = self.quantizer(feats[0])

        # De-duplicate units
        deduplicated_units = [x[0] for x in groupby(units)]

        # units to cjk characters and apply BPE
        cjk_units = "".join([chr(int("4e00", 16) + c) for c in units])
        cjk_tokens = "".join([chr(int("4e00", 16) + c) for c in deduplicated_units])
        bpe_tokens = self.tokenizer.text2tokens(cjk_tokens)
        bpe_tokens = self.converter.tokens2ids(bpe_tokens)
        bpe_tokens = torch.Tensor(bpe_tokens).to(speech.device)

        # Inference using the MT model
        results = self.mt_model(bpe_tokens)

        return {
            "text": results[0][0],
            "units": cjk_units,
            "deduplicated_units": cjk_tokens,
        }

    def state_dict(self, *args, **kwargs):
        return self.clustering_head.state_dict(*args, **kwargs)
