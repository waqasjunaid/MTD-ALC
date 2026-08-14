#!/usr/bin/env python3
"""
flawfinder_detection_baseline.py

Runs Flawfinder (cited as a baseline in LineVul's own MSR'22 paper,
alongside Cppcheck) on the exact same held-out test split used for
MTD+ALC, LineVul, BoW+RF, and Cppcheck, and reports standard detection
metrics.

Threshold selection: uses the VALIDATION set to pick the risk-score
threshold that maximizes F1, then applies that fixed threshold to the
test set -- same methodology as the Cppcheck baseline and your own
MTD+ALC theta*.

Requires flawfinder (already installed via pip earlier in this project):
    pip install flawfinder --break-system-packages

Usage:
    python flawfinder_detection_baseline.py \
        --val_csv data/big-vul_dataset_ours/val.csv \
        --test_csv data/big-vul_dataset_ours/test.csv \
        --out flawfinder_results.json
"""
import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def check_flawfinder_available():
    if shutil.which("flawfinder") is None:
        print("ERROR: 'flawfinder' not found on PATH.")
        print("Install with: pip install flawfinder --break-system-packages")
        sys.exit(1)


def run_flawfinder(code: str, tmpdir: Path, name: str) -> int:
    """Returns the total Flawfinder risk score (sum of hit levels) for
    this function -- same scoring convention as the RQ3 repair-verification
    script, for consistency across this project."""
    path = tmpdir / f"{name}.c"
    path.write_text(code, encoding="utf-8", errors="replace")

    proc = subprocess.run(
        ["flawfinder", "--csv", "--minlevel=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if not proc.stdout.strip():
        return 0

    reader = csv.DictReader(io.StringIO(proc.stdout))
    total_risk = 0
    for row in reader:
        try:
            level = int(row.get("Level", row.get("level", 0)))
        except (ValueError, TypeError):
            continue
        total_risk += level
    return total_risk


def score_dataset(csv_path, tmpdir, cache):
    labels, scores = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            code = row["processed_func"]
            label = int(row["target"])
            key = hash(code)
            if key not in cache:
                cache[key] = run_flawfinder(code, tmpdir, f"tmp_{i}")
            labels.append(label)
            scores.append(cache[key])
    return np.array(labels), np.array(scores)


def metrics_at_threshold(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0
    return {"acc": acc, "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "mcc": mcc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_auc(labels, scores):
    """Rank-based AUC (Mann-Whitney U equivalence), same method used
    elsewhere in this project for MTD+ALC's own AUC."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (float(np.sum(pos_ranks)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    check_flawfinder_available()
    cache = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("Running Flawfinder on validation set (for threshold selection)...")
        val_labels, val_scores = score_dataset(args.val_csv, tmpdir, cache)

        best_threshold, best_f1 = 1, -1
        for t in range(0, int(val_scores.max()) + 1):
            m = metrics_at_threshold(val_labels, val_scores, t)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_threshold = t
        print(f"Best threshold on val set: risk_score >= {best_threshold} "
              f"(val F1={best_f1:.4f})")

        print("Running Flawfinder on test set...")
        test_labels, test_scores = score_dataset(args.test_csv, tmpdir, cache)

    test_metrics = metrics_at_threshold(test_labels, test_scores, best_threshold)
    test_auc = compute_auc(test_labels, test_scores)

    print(f"\n===== Flawfinder baseline test results (threshold={best_threshold}) =====")
    for k in ["acc", "precision", "recall", "f1", "fpr", "mcc"]:
        print(f"  {k}: {test_metrics[k]:.4f}")
    print(f"  auc (threshold-free): {test_auc:.4f}")
    print(f"  (TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']})")

    with open(args.out, "w") as f:
        json.dump({"threshold": best_threshold, "val_f1": best_f1,
                    "test_metrics": test_metrics}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()