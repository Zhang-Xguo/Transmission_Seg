#!/usr/bin/env python3
"""Run Stage B on complete original block LAS files, without scene merging."""
import argparse
import os
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path)
    ap.add_argument("--input-list", type=Path,
                    help="Text file with one input LAS path per line")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--weight", type=Path,
                    help="Checkpoint to use instead of the original Stage B model")
    args = ap.parse_args()
    if args.input_list:
        rows = [x.strip() for x in args.input_list.read_text().splitlines()
                if x.strip() and not x.lstrip().startswith("#")]
        args.inputs = [Path(x.split("\t", 1)[-1]) for x in rows]
        groups = {str(Path(x.split("\t", 1)[-1]).resolve()): x.split("\t", 1)[0]
                  for x in rows if "\t" in x}
    else:
        groups = {}
    if not args.inputs:
        ap.error("provide --inputs or --input-list")
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / "configs/gridnethd/PTv3_gridnethd_7class_stage_b_full_scene_fast.py"
    weight = (args.weight.resolve() if args.weight else
              repo / "exp/gridnethd/ptv3_7class_stage_b_self_finetune/model/model_best.pth")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES=args.gpu,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    for source in args.inputs:
        source = source.resolve()
        name = source.stem
        group = groups.get(str(source), "ungrouped")
        root = args.output_root / group / name
        data = root / "data"
        inference = root / "inference"
        root.mkdir(parents=True, exist_ok=True)
        prep_log = (root / "prepare.log").open("w", encoding="utf-8")
        subprocess.run([args.python, repo / "scripts/prepare_custom_las_ptv3_stream.py",
                        "--input", source, "--output-root", data],
                       check=True, env=env, stdout=prep_log, stderr=subprocess.STDOUT)
        prep_log.close()
        test_log = (root / "inference.log").open("w", encoding="utf-8")
        subprocess.run([args.python, repo / "tools/test.py", "--config-file", cfg,
                        "--num-gpus", "1", "--options", f"save_path={inference}",
                        f"weight={weight}", f"data.test.data_root={data}",
                        "batch_size_test=1", "num_worker=2"],
                       check=True, env=env, stdout=test_log, stderr=subprocess.STDOUT)
        test_log.close()
        metadata = next(data.glob("*_metadata.json"))
        output = root / f"{name}_stage_b_7class_complete_colored.las"
        merge_log = (root / "merge.log").open("w", encoding="utf-8")
        subprocess.run([args.python, repo / "scripts/merge_custom_ptv3.py",
                        "--metadata", metadata, "--data-root", data,
                        "--result-dir", inference / "result", "--output", output,
                        "--num-classes", "7", "--max-missing-fallback", "10"],
                       check=True, env=env, stdout=merge_log, stderr=subprocess.STDOUT)
        merge_log.close()
        print(f"COMPLETE {source} -> {output}", flush=True)


if __name__ == "__main__":
    main()
