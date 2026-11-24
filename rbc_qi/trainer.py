from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import input_channels, load_config, project_path
from .data import RBCQIDataset, build_transform
from .model_factory import create_model
from .tracking import ExperimentLogger, write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def evaluate(model, loader, criterion, device, class_count: int):
    model.eval()
    total_loss = total_correct = total = 0
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            np.add.at(matrix, (labels.cpu().numpy(), predictions.cpu().numpy()), 1)
    precision = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(class_count), where=matrix.sum(axis=0) != 0)
    recall = np.divide(np.diag(matrix), matrix.sum(axis=1), out=np.zeros(class_count), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count), where=(precision + recall) != 0)
    return {"loss": total_loss / total, "accuracy": total_correct / total, "macro_f1": float(f1.mean())}, matrix


def save_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *class_names])
        writer.writerows([[name, *row] for name, row in zip(class_names, matrix)])


def _banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def train(config_path: str | Path, run_dir_override: str | Path | None = None, status_tracker=None) -> Path:
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    experiment_id = config["experiment"]["id"]
    run_dir = Path(run_dir_override) if run_dir_override else project_path(config["project"]["runs_dir"]) / experiment_id
    # The sequential runner creates this directory first to persist the live
    # TRAINING state. Its timestamped run ID prevents collisions.
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_resolved.yaml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    device = choose_device(config["training"]["device"])
    data_cfg = config["data"]
    dataset_root = project_path(data_cfg["dataset_dir"])
    classes = data_cfg["classes"]

    _banner("RBC-QI EfficientNet-B0 Training")
    print(f"Config: {Path(config_path).resolve()}")
    _banner("Configuration")
    print(f"Experiment: {config['experiment']['id']}")
    print(f"Model: {config['model']['name']}")
    print(f"Input features: {config['model']['input_features']}")
    print(f"Number of classes: {len(classes)}")
    print("Classes:")
    for index, name in enumerate(classes):
        print(f"  {index}: {name}")
    print(f"\nDataset: {dataset_root}")
    print(f"Input size: {data_cfg['image_height']} x {data_cfg['image_width']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Weight decay: {config['training']['weight_decay']}")

    _banner("RBC-QI DATASET")
    print(f"Train directory:\n{dataset_root / data_cfg['train_subdir']}")
    print(f"\nTest directory:\n{dataset_root / data_cfg['test_subdir']}")
    train_set = RBCQIDataset(dataset_root / data_cfg["train_subdir"], classes, build_transform(config, training=True))
    test_set = RBCQIDataset(dataset_root / data_cfg["test_subdir"], classes, build_transform(config, training=False))
    print(f"Loaded {len(train_set):,} RBC-QI images from {dataset_root / data_cfg['train_subdir']}")
    print(f"Loaded {len(test_set):,} RBC-QI images from {dataset_root / data_cfg['test_subdir']}")
    workers = min(data_cfg["num_workers"], 8)
    loader_args = {"batch_size": config["training"]["batch_size"], "num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    write_json(run_dir / "dataset_manifest.json", {"train_samples": len(train_set), "test_samples": len(test_set), "classes": classes})

    _banner("DEVICE")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    _banner("CREATING RBC-QI MODEL")
    model = create_model(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    write_json(run_dir / "model_manifest.json", {"input_channels": input_channels(config), "parameters": parameter_count, "model": config["model"]["name"]})
    print("RBC-QI model created successfully.")
    print(f"\nTotal parameters:\n{parameter_count:,}")
    print(f"\nTrainable parameters:\n{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}")
    _banner("RBC-QI CLASSIFIER CHECK")
    print(model.classifier)
    print(f"Classifier output features: {model.classifier[1].out_features}")
    print(f"\nRBC-QI checkpoint directory:\n{run_dir}")

    _banner("RBC-QI FULL FOV FORWARD PASS CHECK")
    model.eval()
    with torch.no_grad():
        dummy_output = model(torch.randn(1, input_channels(config), data_cfg["image_height"], data_cfg["image_width"], device=device))
    print(f"Input shape: {(1, input_channels(config), data_cfg['image_height'], data_cfg['image_width'])}")
    print(f"Output shape: {tuple(dummy_output.shape)}")
    assert tuple(dummy_output.shape) == (1, len(classes))
    print("RBC-QI full FOV forward pass: OK")
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"]["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    logger = ExperimentLogger(config, run_dir, run_dir.name)
    best_accuracy = -1.0
    try:
        for epoch in range(1, config["training"]["epochs"] + 1):
            model.train()
            running_loss = running_correct = seen = 0
            progress = tqdm(train_loader, desc=f"RBC-QI Epoch {epoch}/{config['training']['epochs']}", unit="batch", dynamic_ncols=True)
            for batch_index, (images, labels) in enumerate(progress, start=1):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                running_correct += (logits.argmax(dim=1) == labels).sum().item()
                seen += labels.size(0)
                progress.set_postfix(loss=f"{running_loss / seen:.4f}", acc=f"{running_correct / seen:.4f}")
                if status_tracker is not None and batch_index % config["training"]["progress_every_batches"] == 0:
                    status_tracker.update(
                        "TRAINING", experiment_id=config["experiment"]["id"], epoch=epoch,
                        batch=batch_index, total_batches=len(train_loader),
                    )
            validation, matrix = evaluate(model, test_loader, criterion, device, len(classes))
            metrics = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": running_loss / seen, "train_accuracy": running_correct / seen, "val_loss": validation["loss"], "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"]}
            logger.log_metrics(metrics)
            save_matrix(run_dir / f"confusion_matrix_epoch_{epoch:03d}.csv", matrix, classes)
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "metrics": metrics, "config": config}
            if epoch % config["training"]["save_every_epochs"] == 0:
                torch.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(checkpoint, run_dir / "checkpoint_last.pth")
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                torch.save(checkpoint, run_dir / "checkpoint_best.pth")
            scheduler.step()
            print(f"{run_dir.name}: epoch {epoch}/{config['training']['epochs']} | val_accuracy={validation['accuracy']:.4f} | val_macro_f1={validation['macro_f1']:.4f}", flush=True)
        write_json(run_dir / "summary.json", {"best_val_accuracy": best_accuracy, "epochs": config["training"]["epochs"], "status": "DONE"})
    finally:
        logger.close()
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one RBC-QI experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    arguments = parser.parse_args()
    train(arguments.config, arguments.run_dir)
