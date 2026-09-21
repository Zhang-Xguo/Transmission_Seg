#!/usr/bin/env python3
"""Write an auditable local status report after Stage A exits."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--exit-code", type=int, required=True)
    a = p.parse_args()
    log = a.experiment / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    validations = [tuple(map(float, row)) for row in re.findall(
        r"Val result: mIoU/mAcc/allAcc ([0-9.]+)/([0-9.]+)/([0-9.]+)", text)]
    finished = "Best mIoU:" in text
    best_iou = max((x[0] for x in validations), default=None)
    checkpoint = a.experiment / "model" / "model_best.pth"
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": a.exit_code,
        "completed": a.exit_code == 0 and finished and len(validations) == 10 and checkpoint.exists(),
        "validation_cycles": len(validations),
        "best_miou": best_iou,
        "last_validation": validations[-1] if validations else None,
        "best_checkpoint": str(checkpoint.resolve()) if checkpoint.exists() else None,
        "train_log": str(log.resolve()),
    }
    a.experiment.mkdir(parents=True, exist_ok=True)
    path = a.experiment / "TRAIN_STATUS.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage A status written to {path}: completed={report['completed']}", flush=True)


if __name__ == "__main__":
    main()
