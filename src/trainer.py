import datetime
import json
import os
import time
from glob import glob
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

import datasets
import espnetez as ez
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import psutil
import platform
import yaml
from codecarbon import EmissionsTracker
from espnet2.train.collate_fn import CommonCollateFn
from ptflops import get_model_complexity_info

from src.model import StreamingDSUModel, get_build_model_fn
from src.datasets import ASRDataset, TTSDataset
from src.mcd import compute_mcd


DELTA_DIR = "/scratch/bbjs/shared/corpora"
BASE_DIR = "/home/masao/database"

STATS_DIR = f"exp/stats"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_emissions_and_energy(csv_file_path, num_samples):
    df = pd.read_csv(csv_file_path)
    emissions_sum = df["emissions"].sum()
    energy_consumed_sum = df["energy_consumed"].sum()
    return emissions_sum / num_samples, energy_consumed_sum / num_samples


def log_hardware_info():
    hardware_info = {}
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hardware_info["device_name"] = device_name
        hardware_info["gpu_memory_gb"] = total_memory
        print(f"GPU: {device_name}, Memory: {total_memory:.2f} GB")
    else:
        print("No GPU available.")

    hardware_info["cpu_name"] = platform.processor()
    hardware_info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    hardware_info["cpu_cores_total"] = psutil.cpu_count(logical=True)
    hardware_info["cpu_freq"] = psutil.cpu_freq().current
    print(f"CPU: {hardware_info['cpu_name']}, Cores: {hardware_info['cpu_cores_physical']} physical, {hardware_info['cpu_cores_total']} total, "
          f"Frequency: {hardware_info['cpu_freq']:.2f} MHz")

    svmem = psutil.virtual_memory()
    hardware_info["ram_total_gb"] = svmem.total / (1024 ** 3)
    hardware_info["ram_available_gb"] = svmem.available / (1024 ** 3)
    print(f"RAM: {hardware_info['ram_total_gb']:.2f} GB total, {hardware_info['ram_available_gb']:.2f} GB available")

    return hardware_info


def get_dataset(
    task,
    num_process: int = 1,
    train_split: str = None,
    valid_split: str = None,
    debug: bool = False,
):
    if debug:
        train_dataset = datasets.load_from_disk("debug_data/train")
        valid_dataset = datasets.load_from_disk("debug_data/valid")
        dataset = {"train": train_dataset, "valid": valid_dataset}
        data_info = {
            "speech": lambda x: x["path"]["array"],
            "text": lambda x: np.random.randint(30, size=(1, 10)),
        }
        return dataset, data_info, "train", "valid"

    if task == "asr":
        # Then we will use the following two datasets:
        # - juice500/DSUChallenge2024-wavlm_large-l21-km2000 for discrete units
        # - juice500/DSUChallenge2024 for audio path, text, and ids.

        train_dataset = ASRDataset(split=train_split, num_proc=num_process)
        valid_dataset = ASRDataset(split=valid_split, num_proc=num_process)
        dataset = {train_split: train_dataset, valid_split: valid_dataset}

        data_info = {
            "speech": lambda x: x['audio'],
            "text": lambda x: np.array(x["units"]),
        }
        return dataset, data_info, train_split, valid_split
    elif task == "tts":
        # Then we will use the following two datasets:
        # - juice500/DSUChallenge2024-wavlm_large-l21-km2000 for discrete units
        # - juice500/DSUChallenge2024 for audio path, text, and ids.

        train_dataset = TTSDataset(split=train_split, num_proc=num_process)
        valid_dataset = TTSDataset(split=valid_split, num_proc=num_process)
        dataset = {
            train_split: train_dataset,
            valid_split: valid_dataset
        }

        data_info = {
            "speech": lambda x: x['audio'],
            "text": lambda x: x["text"],
            "sids": lambda x: x["units"],
        }
        return dataset, data_info, train_split, valid_split


class Trainer:
    """Trainer class for handling training, evaluation, and exporting models.

    This class initializes training configurations, datasets, and model paths,
    and provides methods for training, evaluation, quantization, and exporting
    models in various formats (e.g., ONNX).

    Attributes:
        model_config (str): Path to the model configuration file.
        task (str): Task type, e.g., "asr" or "tts".
        dataset (Dict[str, Any]): Dataset dictionary containing "train" and "valid" splits.
        train_dataset (espnetez.dataset.ESPnetEZDataset): Preprocessed training dataset.
        valid_dataset (espnetez.dataset.ESPnetEZDataset): Preprocessed validation dataset.
        exp_dir (Path): Path to the experiment directory.
        data_info (Dict[str, callable]): Data pre-processing information.
        trainer (ez.Trainer): Trainer object for handling the training process.
    """

    def __init__(
        self,
        task: str,
        model_config: str,
        train_args_paths: Union[str, List[str]] = None,
        exp_dir: Union[Path, str] = "exp",
        train_split: str = "train",
        valid_split: str = "valid",
        test_splits: Union[str, List[str]] = "test",
        ngpu: int = 1,
        train: bool = True,
        debug: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        """Initializes the Trainer with task configurations, datasets, and training settings.

        Args:
            task (str): Task type such as "asr" or "tts".
            model_config (str): Path to the model configuration file.
            hf_dataset_or_id (Union[str, Dict[str, Any]]): Hugging Face dataset ID or a dictionary containing preloaded dataset splits.
            train_args_paths (Union[str, List[str]], optional): Path(s) to the training arguments configuration file(s). Defaults to None.
            resume (str, optional): Experiment name to resume training. Defaults to None.
            ngpu (int, optional): Number of GPUs to use. Defaults to 1.
            train (bool, optional): Whether to initialize for training. Defaults to True.
            debug (bool, optional): Whether to use debug dataset. Defaults to False.

        Raises:
            RuntimeError: If the specified resume path does not exist.
            AssertionError: If train is True and train_args_paths is not provided.
        """
        self.model_config = model_config
        self.task = task
        self.exp_dir = exp_dir
        self.sample_rate = sample_rate

        # load configs
        assert train_args_paths is not None, "train_args_paths must be provided"
        if isinstance(train_args_paths, str):
            train_args = ez.from_yaml(task, train_args_paths)
        else:
            train_args = {}
            for paths in train_args_paths:
                train_args = ez.update_finetune_config(task, train_args, paths)

        # initialize training-related attributes
        dataset, data_info, train_key, dev_key = get_dataset(
            task,
            train_args.get("num_process", 1),
            train_split,
            valid_split,
            debug,
        )
        self.dataset = dataset
        self.data_info = data_info
        self.valid_dataset = ez.dataset.ESPnetEZDataset(
            self.dataset[dev_key], data_info=self.data_info
        )

        if train:
            train_args["token_list"] = ["<unk>", "<s>", "</s>", "<pad>"]
            train_args["token_type"] = "char"

            train_dataset = ez.dataset.ESPnetEZDataset(
                self.dataset[train_key], data_info=self.data_info
            )

            self.trainer = ez.Trainer(
                task=task,
                train_config=train_args,
                train_dataset=train_dataset,
                valid_dataset=self.valid_dataset,
                build_model_fn=get_build_model_fn(model_config),
                data_info=self.data_info,
                output_dir=exp_dir,
                stats_dir=STATS_DIR,
                ngpu=ngpu,
            )

    def train(self) -> None:
        """Starts the training process.

        If statistics for the training data are not already collected, it will
        collect the stats before starting training.
        """
        if not os.path.exists(f"{STATS_DIR}/train"):
            print(f"Collecting stats in {STATS_DIR}")
            self.trainer.collect_stats()

        print("Starting training")
        self.trainer.train()

    def _load_ckpt(self, checkpoint_path: str) -> Dict[str, Any]:
        """Loads a checkpoint and removes the "module." prefix from the state dictionary keys.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            Dict[str, Any]: Loaded model state dictionary.
        """
        # load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        d = torch.load(checkpoint_path)
        new_dict = {}
        for keys in d.keys():
            if keys.startswith("module."):  # remove "module." prefix
                new_key = keys[6:]  # remove "module." prefix
                new_dict[new_key] = d[keys]

        model = get_build_model_fn(self.model_config)()
        if len(new_dict.keys()) == len(d.keys()):
            model.load_state_dict(new_dict)
        else:
            model.load_state_dict(d)
        return model

    def evaluate(
        self, checkpoint_path: str = None, model=None, num_workers=1
    ) -> Dict[str, Any]:
        """Evaluates the model on the validation dataset.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (optional): Loaded model to use for evaluation. Defaults to None.

        Returns:
            Dict[str, Any]: Evaluation results.
        """

        if model is None:
            model = get_build_model_fn(self.model_config)()

        # evaluate
        print(f"Evaluating on test data using checkpoint: {checkpoint_path}")
        dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
        )

        hardware_info = log_hardware_info()
        total_params = count_parameters(model)
        print(f"Total parameters: {total_params}")

        total_latency = 0
        n_samples = 0
        gmac_value = None
        first_ten_latency = []
        rtfs = []
        results = []

        tracker = EmissionsTracker(output_dir=self.exp_dir)
        tracker.start()
        for i, batch in tqdm(enumerate(dataloader)):
            # if gmac_value is None:  # computes the FLOPs for the first iteration.
            #     try:
            #         speech = batch[1]["speech"]
            #         speech_lengths = batch[1]["speech_lengths"]
            #         text = batch[1]["text"]
            #         text_lengths = batch[1]["text_lengths"]
            #         input_shape = (speech.shape[1],)

            #         def input_constructor(input_res):
            #             return {
            #                 "speech": torch.randn(1, *input_res).to(speech.device),
            #                 "speech_lengths": torch.tensor([speech_lengths.item()]).to(
            #                     speech.device
            #                 ),
            #                 "text": torch.randint(0, 100, (1, text.shape[1])).to(
            #                     speech.device
            #                 ),
            #                 "text_lengths": torch.tensor([text_lengths.item()]).to(
            #                     speech.device
            #                 ),
            #             }

            #         macs, params = get_model_complexity_info(
            #             model,
            #             input_shape,
            #             input_constructor=input_constructor,
            #             as_strings=True,
            #             print_per_layer_stat=False,
            #             verbose=False,
            #         )
            #         flops_in_gmac = macs.split()[0]
            #         gmac_value = float(flops_in_gmac)
            #         print(f"FLOPs: {gmac_value} GMac (for one inference example)")
            #     except Exception as e:
            #         print(f"Error calculating FLOPs and GMACs: {e}")
            #         gmac_value = str(e)

            start_time = time.time()
            with torch.no_grad():
                out = model.inference(**batch[1])
            latency = time.time() - start_time
            tracker.flush()

            if self.task == "asr":
                results.append(
                    {
                        "hyp": out["text"],
                        "hyp_units": out["units"],
                        "hyp_units_dedup": out["deduplicated_units"],
                        "ref_units": "".join(
                            [chr(int("4e00", 16) + c) for c in batch[1]["text"][0]]
                        ),
                    }
                )

            if self.task == "tts":
                # original_speech = batch[1]["speech"]
                # synthesized_speech = out["wav"]
                # mcd = compute_mcd(original_speech, synthesized_speech)
                results.append(
                    {
                        # "mcd": mcd,
                        "units_fastspeech": out["units_fastspeech"],
                        # "units_hubert": out["units_hubert"],
                        "dedup_fastspeech": out["dedup_fastspeech"],
                        # "dedup_hubert": out["dedup_hubert"],
                        "ref_units": "".join(
                            [chr(int("4e00", 16) + c) for c in batch[1]["sids"][0]]
                        ),
                    }
                )

            # if i < 10:
            #     first_ten_latency.append(latency)

            # if i > 1:
            #     break

            if i > 0:
                total_latency += latency
                n_samples += 1

            if i > 1:  # avoid the first iteration
                rtfs.append(latency / (batch[1]['speech_lengths'].item() / self.sample_rate))

        tracker.stop()
        avg_latency = total_latency / n_samples if n_samples > 0 else 0
        print(f"Average inference latency (after warm-up): {avg_latency:.6f} seconds")
        avg_emissions, avg_energy_consumed = compute_emissions_and_energy(f'{self.exp_dir}/emissions.csv', n_samples+1)

        model._log_stats()

        metrics = {
            "flops_gmac": gmac_value,
            "parameters": total_params,
            "average_latency_sec": avg_latency,
            "average_emissions": avg_emissions,
            "average_energy_consumed": avg_energy_consumed,
            "hardware_info": hardware_info,
            # "first_ten_latency_sec": first_ten_latency,
            "rtf": np.mean(rtfs),
        }

        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        metrics_file_path = Path(self.exp_dir) / "efficiency_metrics.json"
        with open(metrics_file_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Efficiency metrics saved to {metrics_file_path}")

        with open(Path(self.exp_dir) / "results.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {metrics_file_path}")

    def eval_quantize(self, checkpoint_path: str, config: str = None) -> None:
        """Evaluates the model with quantization applied.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            config (str, optional): Path to the quantization configuration file. Defaults to None.

        Raises:
            AssertionError: If `config` is not provided.
        """
        assert config is not None, "quantize_config must be provided"
        model = self._load_ckpt(checkpoint_path)

        # First load a quantization config
        quantize_config = yaml.safe_load(Path(config).read_text())

        if quantize_config.get("dtype", "qint8"):
            dtype = torch.qint8
        elif quantize_config.get("dtype", "float16"):
            dtype = torch.float16

        # Quantize model
        if quantize_config.get("quantize_type", None) == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(model, dtype=dtype)
        elif quantize_config.get("quantize_type", None) == "static":
            model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            quantized_model = torch.quantization.quantize_static(
                model, dtype=dtype, inplace=True
            )

        # Save quantized model
        stem = Path(checkpoint_path).stem
        torch.save(
            {
                "model": quantized_model.state_dict(),
                "quantize_config": quantize_config,
            },
            self.exp_dir / f"{stem}_quantized.pth",
        )
        print(f"Quantized model saved at: {checkpoint_path}")

        # Evaluate quantized model
        self.evaluate(self.exp_dir / f"{stem}_quantized.pth", model)

    def export_onnx(self, checkpoint) -> None:
        """Exports the model to an ONNX format.

        Args:
            checkpoint (str): Path to the checkpoint file.

        Raises:
            AssertionError: If the checkpoint is quantized.
        """
        assert checkpoint is not None, "checkpoint must be provided"
        assert not str(checkpoint).endswith(
            "_quantized.pth"
        ), "checkpoint must not be quantized"

        model = self._load_ckpt(checkpoint)
        model.inference_mode()

        # Prepare input shape
        dummy_input = None
        dummy_lengths = None
        if self.task == "asr":
            dummy_input = torch.randn(1, 8000)
            dummy_lengths = torch.tensor([8000])
            dynamic_axes = {
                "inputs": {1: "length"},
            }
        elif self.task == "tts":
            dummy_input = torch.randn(1, 256)
            dynamic_axes = {
                "inputs": {1: "text_length"},
            }

        # Export ONNX model
        onnx_path = str(checkpoint).replace(".pth", ".onnx")
        print(f"Exporting ONNX model to: {onnx_path}")
        torch.onnx.export(
            model,
            (dummy_input, dummy_lengths),
            onnx_path,
            verbose=True,
            input_names=["inputs", "input_lengths"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
        )
        print("ONNX model exported successfully")
