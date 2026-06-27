#!/usr/bin/env python3
"""
Compute micro frame-level AUC-ROC for Final_Results Stage-3 scores.

Expected layout under --root (default: this script's directory):
  <root>/<DATASET>/video_*.npy          # per-video Stage-3 smoothed scores
  <root>/Ground_Truth/<DATASET>/gt_concat.npy

Per-video predictions are concatenated in numeric video-index order
(video_0000, video_0001, ...) and compared to gt_concat.npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = ("XD-Violence", "UCF-Crime", "MSAD", "ShanghaiTech")
VIDEO_RE = re.compile(r"video_(\d+)\.npy$", re.IGNORECASE)


def load_ravel(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64).ravel()


def sorted_video_npy_files(score_dir: Path) -> List[Path]:
    files: List[Tuple[int, Path]] = []
    for p in score_dir.glob("*.npy"):
        m = VIDEO_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return [p for _, p in sorted(files, key=lambda x: x[0])]


def concat_predictions(score_dir: Path) -> Tuple[np.ndarray, int]:
    files = sorted_video_npy_files(score_dir)
    if not files:
        return np.array([], dtype=np.float64), 0
    parts = [load_ravel(p) for p in files]
    return np.concatenate(parts), len(files)


def micro_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], str]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        return None, f"length mismatch pred={y_score.shape[0]} gt={y_true.shape[0]}"
    if np.unique(y_true).size < 2:
        return None, "only one class in ground truth"
    return float(roc_auc_score(y_true, y_score)), ""


def eval_dataset(root: Path, dataset: str) -> Dict:
    score_dir = root / dataset
    gt_path = root / "Ground_Truth" / dataset / "gt_concat.npy"

    out: Dict = {"dataset": dataset}

    if not score_dir.is_dir():
        out["error"] = f"missing score dir: {score_dir}"
        return out
    if not gt_path.is_file():
        out["error"] = f"missing gt: {gt_path}"
        return out

    y_score, n_videos = concat_predictions(score_dir)
    y_true = load_ravel(gt_path)
    y_bin = (y_true > 0.5).astype(np.float64)

    auc, err = micro_auc(y_bin, y_score) if y_score.size else (None, "no score files")

    out.update(
        {
            "n_videos": n_videos,
            "n_frames": int(y_bin.size),
            "n_positives": int(np.sum(y_bin > 0.5)),
            "pred_frames": int(y_score.size),
            "aucroc": auc,
            "aucroc_pct": round(auc * 100, 2) if auc is not None else None,
            "error": err or None,
        }
    )
    return out


def print_table(results: List[Dict]) -> None:
    print("=" * 72)
    print(f"{'Dataset':<16} {'AUROC %':>10} {'Videos':>8} {'Frames':>12}  Notes")
    print("-" * 72)
    for r in results:
        auc_s = f"{r['aucroc_pct']:.2f}" if r.get("aucroc_pct") is not None else "N/A"
        note = r.get("error") or ""
        print(
            f"{r['dataset']:<16} {auc_s:>10} "
            f"{r.get('n_videos', 0):>8} {r.get('n_frames', 0):>12}  {note}"
        )
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate Final_Results Stage-3 scores vs concatenated GT."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR,
        help="Final_Results root (default: script directory)",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories to evaluate",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/final_aucroc.json)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/final_aucroc.csv)",
    )
    args = ap.parse_args(argv)

    root = args.root.resolve()
    out_json = args.out_json or (root / "final_aucroc.json")
    out_csv = args.out_csv or (root / "final_aucroc.csv")

    results = [eval_dataset(root, ds) for ds in args.datasets]

    payload = {
        "root": str(root),
        "metric": "micro_frame_aucroc",
        "description": "Stage-3 smoothed scores vs gt_concat.npy",
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "aucroc",
                "aucroc_pct",
                "n_videos",
                "n_frames",
                "n_positives",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["dataset"],
                    r.get("aucroc"),
                    r.get("aucroc_pct"),
                    r.get("n_videos"),
                    r.get("n_frames"),
                    r.get("n_positives"),
                    r.get("error") or "",
                ]
            )

    print_table(results)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")

    if any(r.get("aucroc") is None for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
