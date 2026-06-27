#!/usr/bin/env python3
"""
Evaluate micro frame-level AUC-ROC for Table 7 (λ-gating) score exports.

Default layout under --root (script directory):
  <root>/<DATASET>/video_*.npy     # per-video λ-gating scores
  <root>/<DATASET>/gt_concat.npy   # concatenated ground truth (0/1)

Alternatively, pass explicit directories:
  --score-dir /path/to/scores --gt /path/to/gt_concat.npy

GT may also come from:
  --gt-root /path/to/Ground_Truth   # expects <gt-root>/<DATASET>/gt_concat.npy
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


def display_path(path: Path, base: Path) -> str:
    """Return a path relative to base for anonymous/portable outputs."""
    resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        return str(resolved.relative_to(base_resolved))
    except ValueError:
        return path.name


def load_ravel(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64).ravel()


def sorted_video_npy_files(score_dir: Path) -> List[Path]:
    files: List[Tuple[int, Path]] = []
    for p in score_dir.glob("*.npy"):
        if p.name in {"gt_concat.npy", "pred_concat.npy"}:
            continue
        m = VIDEO_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return [p for _, p in sorted(files, key=lambda x: x[0])]


def concat_predictions(score_dir: Path) -> Tuple[np.ndarray, int]:
    files = sorted_video_npy_files(score_dir)
    if files:
        parts = [load_ravel(p) for p in files]
        return np.concatenate(parts), len(files)

    pred_concat = score_dir / "pred_concat.npy"
    if pred_concat.is_file():
        return load_ravel(pred_concat), 0

    return np.array([], dtype=np.float64), 0


def resolve_gt_path(
    dataset: str,
    score_dir: Path,
    gt_root: Optional[Path],
    gt_override: Optional[Path],
) -> Path:
    if gt_override is not None:
        return gt_override
    local_gt = score_dir / "gt_concat.npy"
    if local_gt.is_file():
        return local_gt
    if gt_root is not None:
        return gt_root / dataset / "gt_concat.npy"
    return local_gt


def micro_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], str]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        return None, f"length mismatch pred={y_score.shape[0]} gt={y_true.shape[0]}"
    if np.unique(y_true).size < 2:
        return None, "only one class in ground truth"
    return float(roc_auc_score(y_true, y_score)), ""


def eval_dataset(
    dataset: str,
    score_dir: Path,
    gt_path: Path,
    path_base: Path,
) -> Dict:
    out: Dict = {
        "dataset": dataset,
        "score_dir": display_path(score_dir, path_base),
        "gt_path": display_path(gt_path, path_base),
    }

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
        description="Evaluate Table 7 λ-gating scores vs ground truth."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR,
        help="Table_7 root with one subfolder per dataset (default: script directory)",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories under --root",
    )
    ap.add_argument(
        "--score-dir",
        type=Path,
        default=None,
        help="Single score directory (use with --dataset and --gt)",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name when using --score-dir",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Ground-truth .npy path (gt_concat or per-video concat GT)",
    )
    ap.add_argument(
        "--gt-root",
        type=Path,
        default=None,
        help="Optional GT root: <gt-root>/<DATASET>/gt_concat.npy",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/table7_aucroc.json)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/table7_aucroc.csv)",
    )
    args = ap.parse_args(argv)

    root = args.root.resolve()
    out_json = args.out_json or (root / "table7_aucroc.json")
    out_csv = args.out_csv or (root / "table7_aucroc.csv")
    gt_root = args.gt_root.resolve() if args.gt_root else None

    if args.score_dir is not None:
        if args.dataset is None:
            ap.error("--dataset is required when --score-dir is set")
        score_dir = args.score_dir.resolve()
        gt_path = resolve_gt_path(args.dataset, score_dir, gt_root, args.gt)
        results = [eval_dataset(args.dataset, score_dir, gt_path, root)]
    else:
        results = []
        for ds in args.datasets:
            score_dir = root / ds
            gt_path = resolve_gt_path(ds, score_dir, gt_root, None)
            results.append(eval_dataset(ds, score_dir, gt_path, root))

    payload = {
        "root": ".",
        "metric": "micro_frame_aucroc",
        "description": "Table 7 λ-gating scores vs gt_concat.npy",
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
                "score_dir",
                "gt_path",
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
                    r.get("score_dir"),
                    r.get("gt_path"),
                    r.get("error") or "",
                ]
            )

    print_table(results)
    print(f"\nWrote {display_path(out_json, root)}")
    print(f"Wrote {display_path(out_csv, root)}")

    if any(r.get("aucroc") is None for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
