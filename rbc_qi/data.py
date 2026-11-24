from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class RBCQIDataset(Dataset):
    def __init__(self, root_dir: str | Path, class_names: list[str], transform=None):
        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


class ToFeatureTensor:
    """RGB tensor, optionally extended with a normalized fixed Sobel magnitude map."""
    def __init__(self, feature_mode: str):
        self.feature_mode = feature_mode

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = np.asarray(image, dtype=np.uint8).copy()
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        rgb_tensor = (rgb_tensor - torch.tensor(RGB_MEAN).view(3, 1, 1)) / torch.tensor(RGB_STD).view(3, 1, 1)
        if self.feature_mode == "rgb":
            return rgb_tensor
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(grad_x, grad_y) / (4.0 * 255.0 * np.sqrt(2.0))
        sobel = torch.from_numpy(np.clip(magnitude, 0.0, 1.0)).float().unsqueeze(0)
        return torch.cat((rgb_tensor, sobel), dim=0)


def build_transform(config: dict, training: bool):
    data = config["data"]
    steps = [transforms.Resize((data["image_height"], data["image_width"]))]
    augmentations = data["augmentations"]
    _reject_unsupported_active_augmentations(augmentations)
    if training:
        if augmentations["hflip_p"]:
            steps.append(transforms.RandomHorizontalFlip(p=augmentations["hflip_p"]))
        if augmentations["vflip_p"]:
            steps.append(transforms.RandomVerticalFlip(p=augmentations["vflip_p"]))
        if augmentations["rotate_p"]:
            steps.append(transforms.RandomApply(
                [transforms.RandomRotation(augmentations["rotate_limit_degrees"])],
                p=augmentations["rotate_p"],
            ))
        if augmentations["shift_scale_rotate_p"]:
            scale_limit = augmentations["scale_limit"]
            steps.append(transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=augmentations["rotate_limit_degrees"],
                    translate=(augmentations["shift_limit"], augmentations["shift_limit"]),
                    scale=(1 - scale_limit, 1 + scale_limit),
                )],
                p=augmentations["shift_scale_rotate_p"],
            ))
    steps.append(ToFeatureTensor(config["model"]["input_features"]))
    return transforms.Compose(steps)


def _reject_unsupported_active_augmentations(augmentations: dict) -> None:
    """Never silently apply a YAML option that this repository does not implement."""
    unsupported_probability_keys = (
        "clahe_p", "brightness_contrast_p", "fancy_pca_p", "gauss_noise_p",
        "multiplicative_noise_p", "posterize_p", "rgb_shift_p", "random_gamma_p",
        "random_tone_curve_p", "gaussian_blur_p", "advanced_blur_p", "defocus_p",
        "unsharp_mask_p", "elastic_transform_p", "perspective_p",
        "ringing_overshoot_p", "image_compression_p",
    )
    active = [key for key in unsupported_probability_keys if augmentations.get(key, 0) != 0]
    if augmentations.get("hue_saturation_value", {}).get("prob", 0) != 0:
        active.append("hue_saturation_value.prob")
    if augmentations.get("apply_sharpen_denoise_model", False):
        active.append("apply_sharpen_denoise_model")
    if active:
        raise NotImplementedError(
            "These configured augmentations are not implemented in this QI-safe pipeline: "
            + ", ".join(active)
        )
