#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbc_qi.tracking import mark_interrupted_if_stale

runs_dir = ROOT / "runs"
status_file = runs_dir / "training_status.log"
mark_interrupted_if_stale(runs_dir)
print(status_file.read_text().strip() if status_file.exists() else "No experiment has been started yet.")
