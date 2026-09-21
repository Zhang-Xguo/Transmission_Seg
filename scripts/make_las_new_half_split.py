#!/usr/bin/env python3
"""Make a deterministic spatial half split for las_new tiles."""
import argparse, json, re
from pathlib import Path

def xy(item):
    m = re.search(r"__x(\d+)_y(\d+)$", item)
    if not m:
        raise ValueError(item)
    return int(m.group(1)), int(m.group(2))

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", type=Path, required=True)
ap.add_argument("--seed", type=int, default=20260919)
a = ap.parse_args()
root = a.data_root
all_las = json.loads((root / "val_las_new.json").read_text())
# Split by x coordinate so train and holdout are spatially separated.
xs = sorted({xy(p)[0] for p in all_las})
cut = xs[len(xs) // 2]
train = sorted(p for p in all_las if xy(p)[0] < cut)
holdout = sorted(p for p in all_las if xy(p)[0] >= cut)
if not train or not holdout:
    raise RuntimeError("empty spatial split")
(root / "train_las_new_half.json").write_text(json.dumps(train, indent=2) + "\n")
(root / "val_las_new_holdout.json").write_text(json.dumps(holdout, indent=2) + "\n")
fixed = json.loads((root / "val_fixed.json").read_text())
fixed = [p for p in fixed if not p.startswith("val_final/las_new_target__")]
fixed += holdout
(root / "val_fixed_las_half.json").write_text(json.dumps(sorted(fixed), indent=2) + "\n")
print({"all": len(all_las), "train": len(train), "holdout": len(holdout), "x_cut": cut})
