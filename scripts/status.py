#!/usr/bin/env python3
from pathlib import Path

status_file = Path(__file__).resolve().parents[1] / "runs" / "training_status.log"
print(status_file.read_text().strip() if status_file.exists() else "No experiment has been started yet.")
