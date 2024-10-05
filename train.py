import argparse
import datetime
import logging
import sys
from importlib import import_module
from logging import FileHandler
from pathlib import Path

import espnetez as ez
import torch
import yaml

from src.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--task", type=str, default="asr", help="Task name.")
    parser.add_argument(
        "--model_config", type=str, default=None, help="Yaml config for training."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Yaml config for training. Same format as ESPnet recipes.",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--resume", type=str, default=None, help="Experiment name to resume."
    )
    parser.add_argument(
        "--run_train",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_quantize",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--quantize_config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--log_file", type=str, default="train.log", help="Log file name."
    )
    args = parser.parse_args()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    trainer = Trainer(
        task=args.task,
        model_config=args.model_config,
        hf_dataset_or_id="streaming_dsu",
        train_args_paths=args.train_config,
        resume=args.resume,
        ngpu=args.ngpu,
        debug=args.debug,
    )

    if args.run_train:
        trainer.train()
        if args.ckpt:
            args.ckpt = Path(args.ckpt).parent / "latest.pth"

    if args.evaluate:
        trainer.evaluate(args.ckpt)

    if args.eval_quantize:
        trainer.eval_quantize(args.ckpt, args.quantize_config)

    if args.export_onnx:
        trainer.export_onnx(args.ckpt)
        args.ckpt = Path(args.ckpt).stem + ".onnx"
