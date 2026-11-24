from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import input_channels


def create_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    weights = EfficientNet_B0_Weights.DEFAULT if model_cfg.get("pretrained", False) else None
    model = efficientnet_b0(weights=weights)
    channels = input_channels(config)
    # All non-RGB modes use RGB plus one deterministic fourth feature channel.
    if channels != 3:
        original = model.features[0][0]
        replacement = nn.Conv2d(
            channels, original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=original.bias is not None,
        )
        if weights is not None:
            replacement.weight.data[:, :3].copy_(original.weight.data)
            replacement.weight.data[:, 3:].copy_(original.weight.data.mean(dim=1, keepdim=True))
        model.features[0][0] = replacement
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config["data"]["classes"]))
    return model
