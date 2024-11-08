from parallel_wavegan.models.hifigan import DiscreteSymbolHiFiGANGenerator


import logging

import numpy as np
import torch
from scipy.interpolate import interp1d
import pyworld
import torch.nn as nn
import yaml
from pathlib import Path
import torch.nn.functional as F

"""Pitch Extraction Related"""


def _convert_to_continuous_f0(f0: np.array) -> np.array:
    if (f0 == 0).all():
        logging.warning("All frames seems to be unvoiced.")
        return f0

    # padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nonzero_idxs = np.where(f0 != 0)[0]

    # perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
    f0 = interp_fn(np.arange(0, f0.shape[0]))

    return f0


def f0_dio(
    audio,
    sampling_rate,
    hop_size=160,
    pitch_min=80,
    pitch_max=10000,
    use_log_f0=True,
    use_continuous_f0=True,
):
    """Compute F0 with pyworld.dio

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        hop_size (int): Hop size.
        pitch_min (int): Minimum pitch in pitch extraction.
        pitch_max (int): Maximum pitch in pitch extraction.

    Returns:
        ndarray: f0 feature (#frames, ).

    Note:
        Unvoiced frame has value = 0.

    """
    if torch.is_tensor(audio):
        x = audio.cpu().numpy().astype(np.double)
    else:
        x = audio.astype(np.double)
    frame_period = 1000 * hop_size / sampling_rate

    f0, timeaxis = pyworld.dio(
        x,
        sampling_rate,
        f0_floor=pitch_min,
        f0_ceil=pitch_max,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sampling_rate)
    if use_continuous_f0:
        f0 = _convert_to_continuous_f0(f0)
    if use_log_f0:
        nonzero_idxs = np.where(f0 != 0)[0]
        f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
    return f0
    # if len(f0) > len(mel):
    #     f0 = f0[: len(mel)]
    # else:
    #     f0 = np.pad(f0, (0, len(mel) - len(f0)), mode="edge")


"""Vocoder Related"""


def load_model(checkpoint: Path, config: dict):
    """Load trained model.

    Args:
        checkpoint (Path): Checkpoint path.
        config (dict): Configuration dict.

    Return:
        torch.nn.Module: Model instance.

    """
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = DiscreteSymbolF0Generator(**generator_params)
    model.load_state_dict(
        torch.load(str(checkpoint), map_location="cpu")["model"]["generator"]
    )

    return model


class DiscreteSymbolF0Generator(DiscreteSymbolHiFiGANGenerator):
    """Discrete Symbol HiFiGAN generator module with f0."""

    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        channels=512,
        linear_channel=256,
        num_embs=100,
        num_spk_embs=128,
        spk_emb_dim=128,
        concat_spk_emb=False,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        # discret token
        use_embedding_feats=False,
        use_weight_sum=False,
        layer_num=12,
        use_fix_weight=False,
        use_f0=True,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            num_embs (int): Discrete symbol size
            num_spk_embs (int): Speaker numbers for sPkeaer ID-based embedding
            spk_emb_dim (int): Dimension of speaker embedding
            concat_spk_emb (bool): whether to concat speaker embedding to the input
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_embs=num_embs,
            num_spk_embs=num_spk_embs,
            spk_emb_dim=spk_emb_dim,
            concat_spk_emb=concat_spk_emb,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            use_additional_convs=use_additional_convs,
            bias=bias,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            use_weight_norm=use_weight_norm,
        )

        if use_f0 is True:
            self.use_f0 = use_f0
            self.f0_embedding = torch.nn.Linear(
                in_features=1,
                out_features=linear_channel,
            )

        self.use_weight_sum = use_weight_sum
        if use_weight_sum is True:
            self.layer_num = layer_num
            self.weights = torch.nn.Parameter(torch.ones(self.layer_num))
            self.use_fix_weight = use_fix_weight

            if use_fix_weight is True:  # fix update
                self.weights = torch.nn.Parameter(
                    torch.ones(self.layer_num), requires_grad=False
                )
            else:
                self.weights = torch.nn.Parameter(torch.ones(self.layer_num))

            self.emb = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(
                        num_embeddings=num_embs, embedding_dim=in_channels
                    )
                    for _ in range(self.layer_num)
                ]
            )

        self.input_conv = torch.nn.Conv1d(
            in_channels + linear_channel if use_f0 is True else in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )

        self.use_embedding_feats = use_embedding_feats

    def forward(self, c, f0=None):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor token: (B, 2, T). or (B, 1, T) or (B, L, T)
                        or for embedding feature: (B, C, T)
            f0 (Tensor): Input tensor (B, 1, T)
        Returns:
            Tensor: Output tensor (B, out_channels, T').
        """
        # logging.info(f'feats({c.shape}): {c}')
        # convert idx to embedding
        if self.num_spk_embs > 0:
            assert c.size(1) == 2
            c_idx, g_idx = c.long().split(1, dim=1)
            c = self.emb(c_idx.squeeze(1)).transpose(1, 2)  # (B, C, T)
            g = self.spk_emb(g_idx[:, 0, 0])

            # integrate global embedding
            if not self.concat_spk_emb:
                c = c + g.unsqueeze(2)
            else:
                g = g.unsqueeze(1).expand(-1, c.size(1), -1)
                c = torch.cat([c, g], dim=-1)
        else:
            # NOTE(Yuxun): update for using pretrain model layer output as input
            if not self.use_embedding_feats:
                if self.use_weight_sum:
                    assert c.size(1) == self.layer_num  # (B, L, T)
                    embedded = []
                    for i, embedding_layer in enumerate(self.emb):
                        # Apply the i-th embedding layer to the i-th layer of input
                        embedded.append(embedding_layer(c[:, i].long()))
                    c = torch.stack(embedded, dim=1).transpose(-1, 1)

                    # weights: [L,]
                    if self.use_fix_weight:
                        norm_weights = self.weights
                    else:
                        norm_weights = F.softmax(self.weights, dim=-1)
                    # logging.info(f'norm_weights({norm_weights.shape}): {norm_weights}')
                    # c: (B, C, T, L) * (L,) -> (B, C, T)
                    c = torch.matmul(c, norm_weights)
                else:
                    assert c.size(1) == 1
                    c = self.emb(c.squeeze(1).long()).transpose(1, 2)  # (B, C, T)

        # logging.info(f'f0({f0.shape}): {f0} ')
        if f0 is not None and self.use_f0:
            f0 = self.f0_embedding(f0.transpose(1, 2)).transpose(1, 2)
            c = torch.cat((c, f0), dim=1)
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)
        return c

    def inference(self, c, f0, g=None, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, 2) or (T, 1) or (T, L).
            f0 (Tensor): Input f0 (T,).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        assert not normalize_before, "No statistics are used."
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.long).to(next(self.parameters()).device)
        if g is not None:
            c = c[:, 0:1]
            c = torch.cat([c, c.new_zeros(*c.size()).fill_(g).to(c.device)], dim=1)
        if not self.use_embedding_feats:
            if self.num_spk_embs <= 0 and not self.use_weight_sum:
                c = c[:, 0:1]
        # weight sum: c (T, L)
        c = self.forward(c.transpose(1, 0).unsqueeze(0), f0.unsqueeze(0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class SVSBaseline(nn.Module):
    def __init__(
        self,
        vocoder_checkpoint: Path,
        vocoder_config: Path,
    ):
        super().__init__()
        with open(vocoder_config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        self.f0_predictor = lambda x: f0_dio(
            x,
            sampling_rate=self.config["sampling_rate"],
            hop_size=self.config["hop_size"],
            pitch_max=self.config["sampling_rate"] // 2,
        )
        self.model = load_model(Path(vocoder_checkpoint), self.config)

    def inference(self, singing: torch.Tensor, text: torch.Tensor):
        """
        Args:
            singing (torch.Tensor): (T, ) tensor of the singing audio
            text (torch.Tensor): (L, ) tensor of the discrete tokens
        """
        f0 = self.f0_predictor(singing)
        with torch.no_grad():
            synthesized = self.model.inference(
                text.unsqueeze(1), torch.tensor(f0, dtype=torch.float32)
            )
        return {
            "synthesized": synthesized,
            "singing": singing,
            "units": text,
        }


if __name__ == "__main__":
    import datasets
    import soundfile as sf

    print("Testing SVSBaseline")
    model = SVSBaseline(
        vocoder_checkpoint=Path(
            "/ocean/projects/cis210027p/jhan7/discrete/ParallelWaveGAN/egs/opencpop/token_voc1/exp/train_opencpop_km1024_wavlm_large_layer6/checkpoint-250000steps.pkl"
        ),
        vocoder_config=Path(
            "/ocean/projects/cis210027p/jhan7/discrete/ParallelWaveGAN/egs/opencpop/token_voc1/exp/train_opencpop_km1024_wavlm_large_layer6/config.yml"
        ),
    )
    print("Load dataset")
    split = "test"
    dataset = datasets.load_dataset(
        "jhansss/opencpop_dsu", cache_dir="cache", split=split, streaming=True
    )
    data = next(iter(dataset))
    print("Inference")
    outs = model.inference(
        torch.tensor(data["audio"]["array"].astype(np.float32)),
        torch.tensor(
            np.array(data["token_wavlm_large_1024_6"].split()).astype(np.int64)
        ),
    )
    print("Save synthesized audio")
    sf.write("synthesized.wav", outs["synthesized"].cpu().numpy(), 16000)
