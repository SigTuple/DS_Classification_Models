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
    """Normalized RGB, optionally extended by one deterministic structural feature map."""
    def __init__(self, feature_mode: str, feature_params: dict):
        self.feature_mode = feature_mode
        self.feature_params = feature_params

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = np.asarray(image, dtype=np.uint8).copy()
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
        rgb_tensor = (rgb_tensor - torch.tensor(RGB_MEAN).view(3, 1, 1)) / torch.tensor(RGB_STD).view(3, 1, 1)
        if self.feature_mode == "rgb":
            return rgb_tensor
        feature = extract_feature_map(rgb, self.feature_mode, self.feature_params)
        return torch.cat((rgb_tensor, torch.from_numpy(feature).float().unsqueeze(0)), dim=0)


def extract_feature_map(rgb: np.ndarray, feature_mode: str, feature_params: dict) -> np.ndarray:
    """Return one HxW float32 map in [0, 1] from unnormalized, resized RGB."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if feature_mode == "rgb_sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        feature = np.hypot(grad_x, grad_y) / (4.0 * 255.0 * np.sqrt(2.0))
    elif feature_mode == "rgb_canny":
        params = feature_params["canny"]
        feature = cv2.Canny(
            gray,
            threshold1=params["threshold1"],
            threshold2=params["threshold2"],
            apertureSize=params["aperture_size"],
        ).astype(np.float32) / 255.0
    elif feature_mode == "rgb_local_entropy":
        feature = local_entropy(gray, feature_params["local_entropy"])
    elif feature_mode == "rgb_local_contrast":
        feature = local_contrast(gray, feature_params["local_contrast"])
    elif feature_mode == "rgb_lbp":
        feature = local_binary_pattern(gray, feature_params["lbp"])
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return np.clip(feature, 0.0, 1.0).astype(np.float32)


def local_entropy(gray: np.ndarray, params: dict) -> np.ndarray:
    """Shannon entropy over a configurable square local window using quantized intensities."""
    bins = params["bins"]
    window = params["window_size"]
    if bins < 2 or window < 1 or window % 2 == 0:
        raise ValueError("local_entropy requires bins >= 2 and an odd positive window_size.")
    quantized = np.minimum((gray.astype(np.int32) * bins) // 256, bins - 1)
    entropy = np.zeros(gray.shape, dtype=np.float32)
    for bin_index in range(bins):
        probability = cv2.boxFilter(
            (quantized == bin_index).astype(np.float32),
            ddepth=cv2.CV_32F,
            ksize=(window, window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        term = np.zeros_like(probability)
        valid = probability > 0
        term[valid] = probability[valid] * np.log2(probability[valid])
        entropy -= term
    return entropy / np.log2(bins)


def local_contrast(gray: np.ndarray, params: dict) -> np.ndarray:
    """Local RMS contrast (local standard deviation), scaled to [0, 1]."""
    window = params["window_size"]
    scale = params["normalization_scale"]
    if window < 1 or window % 2 == 0 or scale <= 0:
        raise ValueError("local_contrast requires odd positive window_size and positive normalization_scale.")
    image = gray.astype(np.float32)
    mean = cv2.boxFilter(image, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.boxFilter(image * image, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0)) / scale


def local_binary_pattern(gray: np.ndarray, params: dict) -> np.ndarray:
    """Classic 8-neighbour, radius-1 LBP map normalized from [0, 255] to [0, 1]."""
    if params["points"] != 8 or params["radius"] != 1:
        raise ValueError("This implementation supports LBP with points=8 and radius=1 only.")
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbours = (
        padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
        padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2],
    )
    for bit, neighbour in enumerate(neighbours):
        lbp |= ((neighbour >= center).astype(np.uint8) << bit)
    return lbp.astype(np.float32) / 255.0


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
    steps.append(ToFeatureTensor(config["model"]["input_features"], config["data"]["feature_params"]))
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
