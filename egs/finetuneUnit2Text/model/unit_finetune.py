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
            mt_train_config=ckpt_config,
            mt_model_file=ckpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **kwargs,
        ).mt_model
        self.model.train()

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ):
        """
        speech: (B, T_s)
        text: (B, T_u)
        """
        loss, stats, weight = self.model(
            text=text, text_lengths=text_lengths,
            src_text=src_text, src_text_lengths=src_text_lengths,
        )

        return loss, stats, weight

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        src_text: torch.Tensor,
        src_text_lengths: torch.Tensor,
        **kwargs,
    ):
        return {"feats": src_text, "feats_lengths": src_text_lengths}

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

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
