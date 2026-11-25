#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbc_qi.config import input_channels
from rbc_qi.model_factory import create_model

# Set these paths once, then run this file directly from any directory:
#     python3 /media/sigtupleds/Data/Classification_Model_Training/RBC_QI_Training/scripts/export_onnx.py
# `train.py` is not required. The converter only needs the trained checkpoint
# and the YAML configuration that created that architecture.
DEFAULT_CHECKPOINT_PATH = (
    "/media/sigtupleds/Data/Classification_Model_Training/"
    "RBC_QI_Training/runs/REPLACE_WITH_CHECKPOINT_DIRECTORY/checkpoint_best.pth"
)
DEFAULT_CONFIG_PATH = (
    "/media/sigtupleds/Data/Classification_Model_Training/"
    "RBC_QI_Training/configs/experiments/e1_sobel.yaml"
)
# Default output is written beside DEFAULT_CHECKPOINT_PATH. Change this value
# if a different ONNX filename is required.
DEFAULT_OUTPUT_NAME = "best.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an RBC-QI checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint .pth path. Defaults to DEFAULT_CHECKPOINT_PATH in this file.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Experiment YAML path. Defaults to DEFAULT_CONFIG_PATH in this file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional ONNX output path. Defaults to <checkpoint directory>/DEFAULT_OUTPUT_NAME.",
    )
    arguments = parser.parse_args()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    config_path = Path(arguments.config).resolve()
    if "REPLACE_WITH_CHECKPOINT_DIRECTORY" in str(checkpoint_path):
        raise ValueError(
            "Set DEFAULT_CHECKPOINT_PATH in scripts/export_onnx.py to the trained .pth checkpoint."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    from rbc_qi.config import load_config
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(config)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        # Supports a checkpoint saved as a raw model.state_dict().
        state_dict = checkpoint
    else:
        raise TypeError("Unsupported checkpoint format; expected a PyTorch state dictionary.")
    model.load_state_dict(state_dict)
    model.eval()
    dummy = torch.randn(1, input_channels(config), config["data"]["image_height"], config["data"]["image_width"])
    output = Path(arguments.output).resolve() if arguments.output else checkpoint_path.parent / DEFAULT_OUTPUT_NAME
    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        external_data=False,
    )
    print(f"ONNX written: {output}")


if __name__ == "__main__":
    main()
