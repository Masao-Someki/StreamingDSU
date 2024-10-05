import torch
import torch.nn as nn
from s3prl.nn import Featurizer, S3PRLUpstream


class SampleDistilHubertModel(nn.Module):
    def __init__(
        self,
        output,
        **kwargs,
    ):
        super().__init__()
        self.upstream = S3PRLUpstream(
            "distilhubert",
            path_or_url="./s3prl_model",
            normalize=None,
            extra_conf=None,
        )
        self.upstream.eval()
        self.featurizer = Featurizer(self.upstream, layer_selections=None)
        self.featurizer.eval()
        self.linear = nn.Linear(768, output)
        self.output = output
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        # Your implementation here
        feats, feats_lens = self.upstream(speech, speech_lengths)
        feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])
        out = self.linear(feats).transpose(1, 2)
        out = torch.log_softmax(out, dim=1)
        label = torch.randint(0, self.output, (out.shape[0], out.shape[2])).to(
            out.device
        )

        loss = self.loss(out, label)
        return loss, {
            "loss": loss,
        }

    def inference(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        feats, feats_lens = self.upstream(speech, speech_lengths)
        feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])
        out = self.linear(feats).transpose(1, 2)
        return out
