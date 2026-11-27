#!/usr/bin/env python3
"""Resume the most recent interrupted run for one RBC-QI experiment."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbc_qi.config import project_path
from rbc_qi.tracking import StatusTracker
from rbc_qi.trainer import train


def find_run(runs_dir: Path, experiment_id: str) -> Path:
    matches = [
        path for path in runs_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{experiment_id}") and (path / "checkpoint_last.pth").is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No resumable run for {experiment_id!r}. Expected a run directory with checkpoint_last.pth in {runs_dir}."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume an interrupted RBC-QI experiment from checkpoint_last.pth.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", help="Experiment ID, for example e6_local_contrast.")
    group.add_argument("--run-dir", help="Exact run directory to resume.")
    arguments = parser.parse_args()

    runs_dir = project_path("runs")
    run_dir = Path(arguments.run_dir).resolve() if arguments.run_dir else find_run(runs_dir, arguments.experiment)
    checkpoint = run_dir / "checkpoint_last.pth"
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config required for a safe resume: {config_path}")

    status = StatusTracker(runs_dir, run_dir.name, run_dir)
    status.update("RESUMING", checkpoint=str(checkpoint), config=str(config_path))
    try:
        train(config_path, run_dir, status_tracker=status, resume_checkpoint=checkpoint)
    except KeyboardInterrupt:
        status.update("STOPPED", message="Stopped from terminal while resuming")
        print("\nResume stopped. The latest completed-epoch checkpoint remains available.")
        return 130
    except Exception as error:
        status.update("FAILED", error=str(error))
        traceback.print_exc()
        return 1
    status.update("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
