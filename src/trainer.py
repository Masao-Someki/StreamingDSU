import datetime
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import datasets
import espnetez as ez
import numpy as np
import pandas as pd
import torch
import yaml
from espnet2.train.collate_fn import CommonCollateFn

from src.model import StreamingDSUModel, get_build_model_fn

from ptflops import get_model_complexity_info
from codecarbon import EmissionsTracker

EXP_DIR = "./exp"
STATS_DIR = f"./{EXP_DIR}/stats"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_emissions_and_energy(csv_file_path, num_samples):
    df = pd.read_csv(csv_file_path)
    emissions_sum = df['emissions'].sum()
    energy_consumed_sum = df['energy_consumed'].sum()
    return emissions_sum / num_samples, energy_consumed_sum / num_samples

def log_hardware_info():
    hardware_info = {}
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        hardware_info["device_name"] = device_name
        hardware_info["total_memory_gb"] = total_memory
        print(f"GPU: {device_name}, Memory: {total_memory:.2f} GB")
    else:
        cpu_cores = os.cpu_count()
        hardware_info["device_name"] = "CPU"
        hardware_info["cpu_cores"] = cpu_cores
        print(f"CPU: {cpu_cores} cores")
    
    return hardware_info

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
        hf_dataset_or_id: Union[str, Dict[str, Any]] = None,
        train_args_paths: Union[str, List[str]] = None,
        resume: str = None,
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
        self.sample_rate = sample_rate

        # Experiments name
        if resume is not None and not os.path.exists(f"{EXP_DIR}/{resume}"):
            raise RuntimeError(f"Resume path does not exist: {EXP_DIR}/{resume}")
        elif resume is not None:
            EXPERIMENT_NAME = resume
        else:
            EXPERIMENT_NAME = f"{task}_{hf_dataset_or_id}"

        self.exp_dir = Path(EXP_DIR) / EXPERIMENT_NAME

        # initialize training-related attributes
        if debug:
            train_dataset = datasets.load_from_disk("debug_data/train")
            valid_dataset = datasets.load_from_disk("debug_data/valid")
            self.dataset = {"train": train_dataset, "valid": valid_dataset}
        elif isinstance(hf_dataset_or_id, str):
            self.dataset = datasets.load_dataset(hf_dataset_or_id)
        else:
            self.dataset = hf_dataset_or_id

        self.data_info = {
            "speech": lambda x: x["path"]["array"],
            "text": lambda x: np.random.randint(30, size=(1, 10)),
        }

        self.train_dataset = ez.dataset.ESPnetEZDataset(
            self.dataset["train"], data_info=self.data_info
        )
        self.valid_dataset = ez.dataset.ESPnetEZDataset(
            self.dataset["valid"], data_info=self.data_info
        )

        if train:
            # load configs
            assert train_args_paths is not None, "train_args_paths must be provided"
            if isinstance(train_args_paths, str):
                train_args = ez.from_yaml(task, train_args_paths)
            else:
                train_args = {}
                for paths in train_args_paths:
                    train_args = ez.update_finetune_config(task, train_args, paths)
            train_args["token_list"] = ["<unk>", "<s>", "</s>", "<pad>"]
            train_args["token_type"] = "char"

            self.trainer = ez.Trainer(
                task=task,
                train_config=train_args,
                train_dataset=self.train_dataset,
                valid_dataset=self.valid_dataset,
                build_model_fn=get_build_model_fn(model_config),
                data_info=self.data_info,
                output_dir=f"{EXP_DIR}/{EXPERIMENT_NAME}",
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
            new_key = keys[6:]  # remove "module." prefix
            new_dict[new_key] = d[keys]
        model = get_build_model_fn(self.model_config)()
        model.load_state_dict(new_dict)
        return model

    def evaluate(self, checkpoint_path: str, model=None) -> Dict[str, Any]:
        """Evaluates the model on the validation dataset.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            model (optional): Loaded model to use for evaluation. Defaults to None.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        assert "valid" in self.dataset, "Cannot evaluate on validation data"
        if model is None:
            model = self._load_ckpt(checkpoint_path)

        # evaluate
        print(f"Evaluating on test data using checkpoint: {checkpoint_path}")
        dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.trainer.train_config.num_workers,
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

        tracker = EmissionsTracker(output_dir=self.exp_dir)

        for i, batch in enumerate(dataloader):
            speech = batch[1]["speech"]
            speech_lengths = batch[1]["speech_lengths"]
            text = batch[1]["text"]
            text_lengths = batch[1]["text_lengths"]

            if gmac_value is None: 
                try:
                    input_shape = (speech.shape[1],)  
                    def input_constructor(input_res):
                        return {
                            "speech": torch.randn(1, *input_res).to(speech.device),  
                            "speech_lengths": torch.tensor([speech_lengths.item()]).to(speech.device), 
                            "text": torch.randint(0, 100, (1, text.shape[1])).to(speech.device), 
                            "text_lengths": torch.tensor([text_lengths.item()]).to(speech.device),  
                        }
                    macs, params = get_model_complexity_info(
                        model,
                        input_shape,
                        input_constructor=input_constructor,
                        as_strings=True,
                        print_per_layer_stat=False,
                        verbose=False,
                    )
                    flops_in_gmac = macs.split()[0] 
                    gmac_value = float(flops_in_gmac)
                    print(f"FLOPs: {gmac_value} GMac (for one inference example)")
                except Exception as e:
                    print(f"Error calculating FLOPs and GMACs: {e}")
                    gmac_value = str(e)

            tracker.start()
            start_time = time.time()
            with torch.no_grad():
                model.evaluate(**batch[1])
            latency = time.time() - start_time

            if i < 10: 
                first_ten_latency.append(latency)

            if i > 0: 
                total_latency += latency
                n_samples += 1
            
            if i > 1: # avoid the first iteration
                rtfs.append(latency / (speech_lengths.item() / self.sample_rate))

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
            "first_ten_latency_sec": first_ten_latency,
            "rtf": np.mean(rtfs),
        }

        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        metrics_file_path = Path(self.exp_dir) / "efficiency_metrics.json"
        with open(metrics_file_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Efficiency metrics saved to {metrics_file_path}")


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
