from itertools import groupby

import torch
import torch.nn as nn
from espnet2.bin.tts2_inference import Text2Speech


class TTSBaseline(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        vocoder_path: str,
        vocoder_config_path: str,
        layer: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.model = Text2Speech(
            config_path,
            checkpoint_path,
            vocoder_config=vocoder_config_path,
            vocoder_file=vocoder_path
        )
        self.vocoder = self.model.vocoder
        # self.quantizer = ApplyKmeans(kmeans_path, use_gpu=use_gpu_kmeans)
        # self.tokenizer = self.model.preprocess_fn

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        sid: torch.Tensor,
        sid_lengths: torch.Tensor,
        **kwargs,
    ):
        _ = self.inference(text)
        return None, {}

    def inference(
        self,
        speech: torch.Tensor,
        text: torch.Tensor,
        sids: torch.Tensor,
        **kwargs,
    ):
        # inference with trained model
        out = self.model(text[0])

        # discrete tokens
        discrete_tokens = out['feat_gen'] # Estimated discrete tokens from FastSpeech model
        wav = out['wav'] # Output from vocoder. It will take discrete tokens and convert them into audio

        # generate audio from pre-computed discrete tokens
        wav_hubert = self.vocoder(sids[0].unsqueeze(1))

        # generate discrete tokens from hubert
        # feats, feats_lens = self.model(speech, speech_lengths)
        # units = self.quantizer(feats[0])

        # De-duplicate units
        deduplicated_units = [x[0] for x in groupby(discrete_tokens)]
        # deduplicated_units_hubert = [x[0] for x in groupby(units)]

        return {
            "units_fastspeech": "".join([chr(int("4e00", 16) + c) for c in discrete_tokens]),
            # "units_hubert": "".join([chr(int("4e00", 16) + c) for c in units]),
            "wav": wav,
            "wav_hubert": wav_hubert,
            "dedup_fastspeech": "".join([chr(int("4e00", 16) + c) for c in deduplicated_units]),
            # "dedup_hubert": "".join([chr(int("4e00", 16) + c) for c in deduplicated_units_hubert]),
        }

