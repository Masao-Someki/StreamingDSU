import joblib

import torch
import torch.nn as nn
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.bin.mt_inference import Text2Text
import soundfile as sf
import matplotlib.pyplot as plt
import sentencepiece as spm
from itertools import groupby


class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.C.device)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class WavLMBaselnie(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        kmeans_path: str,
        bpemodel_path: str,
        use_gpu_kmeans: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model = S3prlFrontend(
            fs=16000,
            frontend_conf={
                'upstream': 'wavlm_large',
            },
            download_dir='./hub',
            multilayer_feature=False,
            layer=21,
        )
        self.mt_model = Text2Text(
            mt_train_config=config_path,
            mt_model_file=checkpoint_path,
            beam_size=5
        )
        self.quantizer = ApplyKmeans(kmeans_path, use_gpu=use_gpu_kmeans)
        self.bpemodel = spm.SentencePieceProcessor(model_file=bpemodel_path)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        pass
        return None, None

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        feats, feats_lens = self.model(speech, speech_lengths)
        clusters = self.quantizer(feats[0])

        # De-duplicate clusters
        unique_clusters = [x[0] for x in groupby(clusters)]
        print(unique_clusters)

        # clusters to cjk characters and apply BPE
        cjk_tokens = ''.join([
            chr(int('4e00', 16) + c)
            for c in unique_clusters
        ])
        print(cjk_tokens)
        bpe_tokens = self.bpemodel.encode(cjk_tokens)
        bpe_tokens = torch.Tensor(bpe_tokens).to(speech.device)
        print(bpe_tokens)
        results = self.mt_model(bpe_tokens)
        print(results)
        import sys; sys.exit()
        
        return
