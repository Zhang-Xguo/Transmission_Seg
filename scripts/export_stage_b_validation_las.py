#!/usr/bin/env python3
"""Export tiled Pointcept predictions and references as scene-level colored LAS."""
import argparse
import json
from pathlib import Path

import laspy
import numpy as np


NAMES = ["tower", "conductor", "insulator", "vegetation",
         "building", "ground", "other"]
PALETTE8 = np.asarray([
    [255, 0, 0],       # tower: red
    [255, 215, 0],     # conductor: yellow/gold
    [255, 0, 255],     # insulator: magenta
    [0, 180, 0],       # vegetation: green
    [0, 100, 255],     # building: blue
    [160, 100, 50],    # ground: brown
    [150, 150, 150],   # other: gray
], dtype=np.uint16)


def make_header(mins):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    header.offsets = np.floor(mins)
    header.add_extra_dim(laspy.ExtraBytesParams(name="classif", type=np.uint8,
                                                 description="seven-class label"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="reference7", type=np.uint8,
                                                 description="seven-class reference"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="label_source", type=np.uint8,
                                                 description="1 manual, 0 pseudo"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="sensor_red", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="sensor_green", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="sensor_blue", type=np.uint8))
    return header


def write_tile(writer, tile_dir, labels, reference, scene_mins):
    coord = np.load(tile_dir / "coord.npy")
    color = np.load(tile_dir / "color.npy").astype(np.uint8)
    source = np.load(tile_dir / "label_source.npy").reshape(-1).astype(np.uint8)
    labels = labels.reshape(-1).astype(np.uint8)
    reference = reference.reshape(-1).astype(np.uint8)
    if not (len(coord) == len(labels) == len(reference) == len(source)):
        raise RuntimeError(f"Point-count mismatch for {tile_dir}")
    points = laspy.ScaleAwarePointRecord.zeros(len(coord), header=writer.header)
    xyz = coord.astype(np.float64) + scene_mins
    points.x, points.y, points.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    points.classification = labels
    points.classif = labels
    points.reference7 = reference
    points.label_source = source
    points.sensor_red, points.sensor_green, points.sensor_blue = color.T
    rgb = PALETTE8[labels] * 257
    points.red, points.green, points.blue = rgb.T
    writer.write_points(points)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_root / "collection_manifest.json").read_text(encoding="utf-8"))
    records = [record for record in manifest if record["split"] == "val_final"]
    summary = {"classes": {str(i): {"name": name, "rgb8": PALETTE8[i].tolist()}
                           for i, name in enumerate(NAMES)}, "scenes": {}}

    for record in records:
        scene = record["scene"]
        report = json.loads(
            (data_root / f"{scene}_val_final_report.json").read_text(encoding="utf-8")
        )
        with laspy.open(record["original_las"][0]) as source:
            scene_mins = np.asarray(source.header.mins, dtype=np.float64)
        for raw_path in record["original_las"][1:]:
            with laspy.open(raw_path) as source:
                scene_mins = np.minimum(scene_mins, source.header.mins)

        pred_path = output_root / f"{scene}_stage_b_prediction_colored.las"
        ref_path = output_root / f"{scene}_reference_colored.las"
        header = make_header(scene_mins)
        counts = np.zeros(7, dtype=np.int64)
        with laspy.open(pred_path, mode="w", header=header) as pred_writer, \
                laspy.open(ref_path, mode="w", header=make_header(scene_mins)) as ref_writer:
            for tile in report["tiles"]:
                tile_dir = data_root / "val_final" / tile["tile"]
                reference = np.load(tile_dir / "segment.npy")
                prediction = np.load(prediction_root / f"{tile['tile']}_pred.npy")
                write_tile(pred_writer, tile_dir, prediction, reference, scene_mins)
                write_tile(ref_writer, tile_dir, reference, reference, scene_mins)
                counts += np.bincount(prediction.reshape(-1), minlength=7)[:7]
        summary["scenes"][scene] = {
            "prediction_las": str(pred_path), "reference_las": str(ref_path),
            "tiles": len(report["tiles"]), "points": int(counts.sum()),
            "prediction_counts": {NAMES[i]: int(counts[i]) for i in range(7)},
        }
        print(f"EXPORTED {scene}: {counts.sum()} points", flush=True)

    (output_root / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE: {output_root}")


if __name__ == "__main__":
    main()
