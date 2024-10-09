import argparse
from argparse import Namespace
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List

import espnetez as ez
import numpy as np
import soundfile as sf
import torch
import yaml
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.dataset import kaldi_loader


class StreamingDSUModel(AbsESPnetModel):
    """A streaming version of a DSU (Deep Structured Understanding) model.

    This class is designed to handle both training and inference modes for streaming tasks.
    It logs statistics during training and supports flexible feature extraction and forward methods.

    Attributes:
        config (Namespace): Parsed configuration parameters.
        model (torch.nn.Module): The internal model built using the provided configuration.
        model_name (str): The name of the model derived from the configuration.
        log_interval (int): Interval for logging statistics during training.
        log_stats (Dict[str, List[float]]): Dictionary to store logged statistics.
        mode (str): Current mode of the model, either "train" or "inference".
    """

    def __init__(self, train_config: dict) -> None:
        """Initializes the StreamingDSUModel with the given training configuration.

        Args:
            train_config (dict): Dictionary containing the model configuration and parameters.
        """
        super().__init__()
        module = train_config["module"]
        self.config = Namespace(**train_config)
        clz = import_class(module)
        self.model = clz(**train_config)
        self.model_name = module.split(".")[-1]
        self.log_interval = (
            hasattr(train_config, "log_interval")
            and train_config["log_interval"]
            or 100
        )
        self.log_stats = {}
        self.mode = "train"
        self.internal_counter = 0

    def inference_mode(self) -> None:
        """Sets the model to inference mode."""
        self.mode = "inference"

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collects and processes features from the input speech tensor.

        Args:
            speech (torch.Tensor): Input speech tensor.
            speech_lengths (torch.Tensor): Lengths of each speech sequence in the batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing processed features and their lengths.
        """
        return {"feats": speech, "feats_lengths": speech_lengths}

    def _log_stats(self) -> None:
        """Logs the accumulated statistics."""
        text = ""
        self.internal_counter = 0
        for key in self.log_stats:
            if isinstance(self.log_stats[key][0], torch.Tensor):
                self.log_stats[key] = [
                    el.detach().cpu().numpy() for el in self.log_stats[key]
                ]

            avg = np.mean(self.log_stats[key])
            std = np.std(self.log_stats[key])
            text += f"{key}: {avg:.3f} ({std:.3f})  "
            self.log_stats[key] = []
        print(text)

    def _aggregate_log_stats(self, stats: Dict[str, float]) -> None:
        """Aggregates the statistics for logging.

        Args:
            stats (Dict[str, float]): Dictionary containing current batch statistics.
        """
        self.internal_counter += 1
        for key, value in stats.items():
            if key not in self.log_stats:
                self.log_stats[key] = []
            self.log_stats[key].append(value)

    def forward(self, *args, **kwargs):
        """Defines the forward pass of the model.

        Depending on the mode (train or inference), the method either computes loss and logs
        training statistics or performs inference on the input data.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float], None]]:
            - If in training mode: returns loss, statistics, and None.
            - If in inference mode: returns the inference results from the model.
        """
        if self.mode == "train":
            loss, stats = self.model(*args, **kwargs)
            self._aggregate_log_stats(stats)
            if self.internal_counter % self.log_interval == 0:
                self._log_stats()
            return loss, stats, None
        elif self.mode == "inference":
            return self.model.inference(*args, **kwargs)

    def inference(self, *args, **kwargs) -> None:
        """Evaluates the model on the provided inputs.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        return self.model.inference(*args, **kwargs)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state dictionary into the model.

        Args:
            state_dict (Dict[str, Any]): Dictionary containing model state.
        """
        self.model.load_state_dict(state_dict)


def import_class(module: str):
    """Imports a class dynamically from a given module path.

    Args:
        module (str): The full module path as a string.

    Returns:
        type: The class object referred to by the module path.
    """
    class_name = module.split(".")[-1]
    module_name = ".".join(module.split(".")[:-1])
    m = import_module(module_name)
    return getattr(m, class_name)


def get_build_model_fn(model_config: str):
    """Builds a function to create a StreamingDSUModel instance.

    Args:
        model_config (str): Path to the model configuration file.

    Returns:
        Callable: Function that constructs and returns a StreamingDSUModel.
    """
    with open(model_config, "r") as f:
        model_config = yaml.safe_load(f)

    def build_model_fn(*args, **kwargs) -> StreamingDSUModel:
        """Instantiates the StreamingDSUModel.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            StreamingDSUModel: Constructed model.
        """
        return StreamingDSUModel(model_config)

    return build_model_fn
