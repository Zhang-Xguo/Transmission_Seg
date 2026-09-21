#!/usr/bin/env python3
"""Run identical LAS inference with the public baseline and a fine-tuned weight."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def logged(command, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        subprocess.run([str(x) for x in command], check=True, stdout=stream,
                       stderr=subprocess.STDOUT, env=env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--baseline-weight", required=True)
    p.add_argument("--finetuned-weight", required=True)
    p.add_argument("--gpu", default="4")
    p.add_argument("--python", default=(
        sys.executable))
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    src, root = Path(a.input).resolve(), Path(a.output_root).resolve()
    scene = root / src.stem
    data = scene / "data"
    metadata = data / f"{src.stem}_metadata.json"
    if not metadata.exists():
        logged([a.python, repo / "scripts/prepare_custom_las_ptv3_stream.py",
                "--input", src, "--output-root", data], scene / "prepare.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = a.gpu
    env["PYTHONPATH"] = str(repo)
    outputs = {}
    for tag, weight in (("baseline", a.baseline_weight),
                        ("finetuned", a.finetuned_weight)):
        exp = scene / f"exp_{tag}"
        result = exp / "result"
        logged([a.python, repo / "tools/test.py", "--config-file",
                repo / "configs/gridnethd/PTv3_gridnethd_color_fast.py",
                "--num-gpus", "1", "--options", f"save_path={exp}",
                f"weight={Path(weight).resolve()}", f"data.test.data_root={data}",
                "batch_size_test=1", "num_worker=8"], scene / f"{tag}.log", env)
        output = scene / f"{src.stem}_{tag}_colored.las"
        logged([a.python, repo / "scripts/merge_custom_ptv3.py", "--metadata",
                metadata, "--data-root", data, "--result-dir", result,
                "--output", output, "--max-missing-fallback", "10"],
               scene / f"{tag}_merge.log")
        outputs[tag] = str(output)
    summary = {"input": str(src), "baseline_weight": str(Path(a.baseline_weight).resolve()),
               "finetuned_weight": str(Path(a.finetuned_weight).resolve()),
               "outputs": outputs, "same_preprocessed_input": True}
    (scene / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
