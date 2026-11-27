from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


class StatusTracker:
    def __init__(self, runs_dir: Path, run_id: str, run_dir: Path):
        self.runs_dir = runs_dir
        self.run_id = run_id
        self.run_dir = run_dir

    def update(self, state: str, **details: Any) -> None:
        now = utc_now()
        payload = {
            "run_id": self.run_id,
            "state": state,
            "updated_at": now,
            "heartbeat_at": now,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            **details,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "status.json", payload)
        line = " | ".join(f"{key}={value}" for key, value in payload.items()) + "\n"
        temp = self.runs_dir / ".training_status.tmp"
        temp.write_text(line)
        os.replace(temp, self.runs_dir / "training_status.log")


def mark_interrupted_if_stale(runs_dir: Path) -> dict[str, Any] | None:
    """Mark a locally dead TRAINING process interrupted when status is inspected."""
    status_file = runs_dir / "training_status.log"
    if not status_file.exists():
        return None
    fields = dict(item.split("=", 1) for item in status_file.read_text().strip().split(" | ") if "=" in item)
    if fields.get("state") not in {"TRAINING", "PREPARING_CACHE", "RESUMING"}:
        return fields
    if fields.get("hostname") != socket.gethostname():
        fields["liveness"] = "unknown (status belongs to another host)"
        return fields
    try:
        os.kill(int(fields.get("pid", "-1")), 0)
    except (ProcessLookupError, ValueError):
        run_dir = runs_dir / fields["run_id"]
        tracker = StatusTracker(runs_dir, fields["run_id"], run_dir)
        tracker.update(
            "INTERRUPTED",
            experiment_id=fields.get("experiment_id"),
            message="Training process is no longer running; resume from checkpoint_last.pth.",
            previous_heartbeat_at=fields.get("heartbeat_at"),
        )
        return {"state": "INTERRUPTED", "run_id": fields["run_id"]}
    except PermissionError:
        fields["liveness"] = "unknown (permission denied checking PID)"
    else:
        fields["liveness"] = "running"
    return fields


class ExperimentLogger:
    def __init__(self, config: dict, run_dir: Path, run_id: str):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.mlflow = None
        mlflow_cfg = config["tracking"].get("mlflow", {})
        if mlflow_cfg.get("enabled", False):
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
                mlflow.set_experiment(mlflow_cfg["experiment_name"])
                self.mlflow = mlflow
                mlflow.start_run(run_name=run_id)
                mlflow.log_params({
                    "experiment_id": config["experiment"]["id"],
                    "input_features": config["model"]["input_features"],
                    "batch_size": config["training"]["batch_size"],
                    "epochs": config["training"]["epochs"],
                })
            except Exception as error:
                print(f"WARNING: MLflow disabled because it could not start: {error}")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        if self.mlflow is not None:
            self.mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}, step=int(metrics["epoch"]))

    def close(self) -> None:
        if self.mlflow is not None:
            self.mlflow.end_run()
