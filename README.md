# SAAF — Anonymous Score Release (Review Period)

This repository contains **pre-computed frame-level anomaly scores** and **evaluation scripts** for reproducing reported **micro frame-level AUC-ROC** results of **SAAF** on four benchmarks:

- **XD-Violence**
- **UCF-Crime**
- **MSAD**
- **ShanghaiTech**

All paths in evaluation outputs are written **relative to each experiment folder** (no host-specific absolute paths).

---

## Repository layout

```
.
├── Final_Results/          # Full SAAF pipeline (Stage 4)
│   ├── eval_final_aucroc.py
│   ├── final_aucroc.json
│   ├── final_aucroc.csv
│   ├── Ground_Truth/       # Concatenated GT per dataset
│   │   └── <DATASET>/gt_concat.npy
│   └── <DATASET>/          # Per-video scores: video_0000.npy, video_0001.npy, ...
│
├── Ablations/
│   ├── Table_7/            # Stage-wise ablation: λ-gating
│   │   ├── eval_table7_aucroc.py
│   │   ├── table7_aucroc.json
│   │   └── <DATASET>/      # video_*.npy + gt_concat.npy
│   │
│   ├── Table_8/            # Refinement ablation: Stage-2 frame scores + smoothing
│   │   ├── eval_table8_aucroc.py
│   │   ├── export_table8_stage2_smoothing.py
│   │   ├── table8_aucroc.json
│   │   └── <DATASET>/      # video_*.npy + gt_concat.npy
│   │
│   └── export_table8_stage2_smoothing.py
│
└── Runs/                   # Qwen seed-variance runs (MSAD, UCF_Crime, XD_Violence)
    └── <DATASET>/seed_<N>/eval_ffill/
```

### Per-dataset folder contents

Each `<DATASET>/` directory under `Final_Results`, `Ablations/Table_7`, or `Ablations/Table_8` contains:

| File | Description |
|------|-------------|
| `video_*.npy` | Per-video frame-level anomaly scores (float32), indexed `video_0000`, `video_0001`, … |
| `gt_concat.npy` | Concatenated binary ground truth (0/1) aligned with concatenated predictions |
| `pred_concat.npy` | (Ablations only) Pre-concatenated predictions for quick loading |
| `metrics.json` | (Ablations only) Export metadata and verified AUC-ROC |

### `Runs/` — Qwen seed-variance outputs

Raw **Stage-2 Qwen-VL scoring runs** used for the seed-stability analysis (seeds **42**, **123**, **456**). Currently included for **MSAD**, **UCF_Crime**, and **XD_Violence** (~3 GB).

```
Runs/
└── <DATASET>/
    └── seed_<42|123|456>/
        ├── eval_ffill/
        │   ├── metrics.json          # micro AUC-ROC for this seed
        │   ├── gt_concat.npy
        │   ├── ffill_pred_concat.npy
        │   └── per_video/            # per-video forward-filled score .npy
        └── …                         # per-frame Qwen JSON outputs (by video)
```

Check `eval_ffill/metrics.json` under each seed for the reported AUC-ROC. Mean ± std across the three seeds (paper): MSAD **78.70%**, UCF **66.53%**, XD **83.99%**.

---

## Requirements

- Python 3.8+
- `numpy`
- `scikit-learn`

Optional (only to **re-export** Table 8 scores from upstream frame scores):

- `scipy`

Install:

```bash
pip install numpy scikit-learn scipy
```

---

## How to reproduce results

Clone the repository and run the evaluators from the corresponding folder. Each script defaults to **`.`** (its own directory) as `--root`.

### 1. Final SAAF results 

Corresponds to the **bold final row** in the main results table.

```bash
cd Final_Results
python3 eval_final_aucroc.py
```

**Expected AUC-ROC (%):**

| Dataset | AUC-ROC |
|---------|--------:|
| XD-Violence | 92.07 |
| UCF-Crime | 82.59 |
| MSAD | 86.70 |
| ShanghaiTech | 80.54 |

Outputs: `final_aucroc.json`, `final_aucroc.csv`

GT is read from `Ground_Truth/<DATASET>/gt_concat.npy` when present; each dataset folder may also include a local `gt_concat.npy` copy.

---

### 2. Table 7 — Stage-wise ablation (+ λ-gating)

Pre-exported **mild λ-fusion** scores (λ_δ=0.6, λ_ρ=0.4, λ_both=0.5, τ_f=0.4).

```bash
cd Ablations/Table_7
python3 eval_table7_aucroc.py
```

**Paper-reported AUC-ROC (%):**

| Dataset | AUC-ROC |
|---------|--------:|
| XD-Violence | 88.53 |
| UCF-Crime | 76.27 |
| MSAD | 82.36 |
| ShanghaiTech | 75.50 |

Outputs: `table7_aucroc.json`, `table7_aucroc.csv`

> **Note:** Scores stored as float32 may round to slightly different values (e.g. 88.56 vs 88.53) depending on concatenation order. the paper values were computed in float64 during the original ablation run.

---

### 3. Table 8 — Refinement ablation (Stage 2 + smoothing)

Stage-1 OR-gated frame scores smoothed with `gaussian_filter1d` per video:

- XD / UCF / MSAD: **σ = 100**
- ShanghaiTech: **σ = 30**

**Evaluate bundled scores:**

```bash
cd Ablations/Table_8
python3 eval_table8_aucroc.py
```

**Expected AUC-ROC (%):**

| Dataset | AUC-ROC |
|---------|--------:|
| XD-Violence | 89.35 |
| UCF-Crime | 73.30 |
| MSAD | 81.78 |
| ShanghaiTech | 77.19 |

Outputs: `table8_aucroc.json`, `table8_aucroc.csv`

**Re-export** (requires access to upstream SAAF frame-score directories outside this repo):

```bash
cd Ablations
python3 export_table8_stage2_smoothing.py
```

Then re-run `eval_table8_aucroc.py` inside `Table_8/`.

---

## Evaluation method

All scripts compute **micro frame-level AUC-ROC**:

1. Load `gt_concat.npy` (or `Ground_Truth/<DATASET>/gt_concat.npy`).
2. Concatenate per-video `video_*.npy` files in numeric index order (`video_0000`, `video_0001`, …).
3. Binarize ground truth: `gt > 0.5`.
4. Compute `sklearn.metrics.roc_auc_score`.

Single-dataset evaluation example:

```bash
python3 eval_table7_aucroc.py \
  --score-dir ./MSAD \
  --dataset MSAD \
  --gt ./MSAD/gt_concat.npy
```

---

## Mapping to paper tables

| Paper table | Folder | Configuration |
|-------------|--------|-----------------|
| Main results (Ours) | `Final_Results/` | Stage 3: λ-gated fusion + Gaussian refinement |
| Stage-wise ablation (λ-gating row) | `Ablations/Table_7/` | Mild λ-fusion, no final refinement |
| Refinement ablation (smoothing row) | `Ablations/Table_8/` | Stage-1 frame scores + Gaussian smoothing only |
| Qwen seed variance | `Runs/` | Three random seeds per dataset; see `eval_ffill/metrics.json` |

---

## Datasets summary

| Dataset | Videos | Frames (approx.) |
|---------|-------:|-----------------:|
| XD-Violence | 800 | 2,335,801 |
| UCF-Crime | 290 | 1,111,808 |
| MSAD | 360 | 227,524 |
| ShanghaiTech | 107 | 40,791 |

---

##  Review use

This release is intended **solely for anonymous peer review** of SAAF. Scores are frame-level `.npy` arrays; raw videos and full training code are not included here.

