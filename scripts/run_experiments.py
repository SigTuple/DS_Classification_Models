#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIGS = [
    str(ROOT / "configs" / "experiments" / "e0_baseline.yaml"),
    str(ROOT / "configs" / "experiments" / "e1_sobel.yaml"),
    str(ROOT / "configs" / "experiments" / "e2_canny_edge_map.yaml"),
    str(ROOT / "configs" / "experiments" / "e3_downscaled_batch2.yaml"),
    str(ROOT / "configs" / "experiments" / "e4_downscaled_batch4.yaml"),
    str(ROOT / "configs" / "experiments" / "e5_local_entropy.yaml"),
    str(ROOT / "configs" / "experiments" / "e6_local_contrast.yaml"),
    str(ROOT / "configs" / "experiments" / "e7_lbp.yaml"),
    str(ROOT / "configs" / "experiments" / "e8_full_fov_batch4.yaml"),
]

from rbc_qi.config import load_config, project_path
from rbc_qi.tracking import StatusTracker
from rbc_qi.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RBC-QI experiment configs sequentially.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Experiment YAML files, in run order. Defaults to E0 through E8.",
    )
    arguments = parser.parse_args()
    for config_path in arguments.configs:
        config = load_config(config_path)
        runs_dir = project_path(config["project"]["runs_dir"])
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['experiment']['id']}"
        run_dir = runs_dir / run_id
        status = StatusTracker(runs_dir, run_id, run_dir)
        status.update("TRAINING", experiment_id=config["experiment"]["id"], config=str(Path(config_path).resolve()))
        try:
            train(config_path, run_dir, status_tracker=status)
        except KeyboardInterrupt:
            status.update("STOPPED", experiment_id=config["experiment"]["id"], message="Stopped from terminal")
            print("\nTraining stopped from terminal. Status written as STOPPED.")
            return 130
        except Exception as error:
            status.update("FAILED", experiment_id=config["experiment"]["id"], error=str(error))
            traceback.print_exc()
            return 1
        status.update("DONE", experiment_id=config["experiment"]["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
