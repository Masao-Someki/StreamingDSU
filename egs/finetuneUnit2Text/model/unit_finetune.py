import torch
import torch.nn as nn

from espnet2.bin.mt_inference import Text2Text
from espnet2.train.abs_espnet_model import AbsESPnetModel


class MTModel(AbsESPnetModel):
    def __init__(
        self,
        ckpt_config: str,
        ckpt_path: str,
        **kwargs,
    ):
        super().__init__()
        self.model = Text2Text(
            ckpt_config,
            ckpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **kwargs,
        ).mt_model
        self.model.train()


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
        loss, stats, weight = self.model(
            src_text=speech, src_text_lengths=speech_lengths,
            text=text, text_lengths=text_lengths,
        )

        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": speech, "feats_lengths": speech_lengths}

    # def state_dict(self, *args, **kwargs):
    #     return {
    #         "model": self.model.state_dict(),
    #     }
    
    # def load_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict["model"])

    # def inference(
    #     self,
    #     speech: torch.Tensor,
    #     **kwargs,
    # ):
    #     assert len(speech) == 1
    #     speech_lengths = torch.LongTensor([len(speech[0])]).to(speech.device)
    #     with torch.no_grad():
    #         feats, feats_lens = self.model.encode(speech, speech_lengths) # (B, T, D)


    #     units = self.clustering_head(feats) # (B, T, Cluster)
    #     return units.argmax(dim=-1)[0]
