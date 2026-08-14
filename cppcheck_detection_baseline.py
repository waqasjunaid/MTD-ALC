#!/usr/bin/env python3
"""
cppcheck_detection_baseline.py

Runs cppcheck (a static analyzer cited as a baseline in LineVul's own
MSR'22 paper) on the exact same held-out test split used for MTD+ALC
and the LineVul retraining, and reports standard detection metrics.

Threshold selection: uses the VALIDATION set (not test) to pick the
warning-count threshold that maximizes F1, then applies that fixed
threshold to the test set -- same methodology already used for your
own MTD+ALC threshold (theta*), for a fair comparison.

Requires cppcheck installed:
    sudo apt-get install cppcheck
    # or: conda install -c conda-forge cppcheck

Usage:
    python cppcheck_detection_baseline.py \
        --train_csv data/big-vul_dataset_ours/train.csv \
        --val_csv data/big-vul_dataset_ours/val.csv \
        --test_csv data/big-vul_dataset_ours/test.csv \
        --out cppcheck_results.jsonl
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def check_cppcheck_available():
    if shutil.which("cppcheck") is None:
        print("ERROR: 'cppcheck' not found on PATH.")
        print("Install with: sudo apt-get install cppcheck")
        sys.exit(1)


def run_cppcheck(code: str, tmpdir: Path, name: str) -> int:
    """Returns the number of cppcheck warnings for this function."""
    path = tmpdir / f"{name}.c"
    path.write_text(code, encoding="utf-8", errors="replace")
    proc = subprocess.run(
        ["cppcheck", "--enable=warning,style,performance,portability",
         "--suppress=missingIncludeSystem", str(path)],
        capture_output=True, text=True, timeout=15,
    )
    # cppcheck writes findings to stderr, one per line, prefixed with the filename
    warnings = [l for l in proc.stderr.splitlines() if str(path) in l and ":" in l]
    return len(warnings)


def score_dataset(csv_path, tmpdir, cache, ids_out=None):
    """Returns (labels, warning_counts) arrays, using a cache keyed by
    code content to avoid re-running cppcheck on identical functions.
    If ids_out is given, also appends each row's index (or id column,
    if present) to it, for later per-sample comparison."""
    labels, counts = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            code = row["processed_func"]
            label = int(row["target"])
            key = hash(code)
            if key not in cache:
                cache[key] = run_cppcheck(code, tmpdir, f"tmp_{i}")
            labels.append(label)
            counts.append(cache[key])
            if ids_out is not None:
                ids_out.append(row.get("id", i))
    return np.array(labels), np.array(counts)


def metrics_at_threshold(labels, counts, threshold):
    preds = (counts >= threshold).astype(int)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    check_cppcheck_available()
    cache = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("Running cppcheck on validation set (for threshold selection)...")
        val_labels, val_counts = score_dataset(args.val_csv, tmpdir, cache)

        best_threshold, best_f1 = 1, -1
        for t in range(0, int(val_counts.max()) + 1):
            m = metrics_at_threshold(val_labels, val_counts, t)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_threshold = t
        print(f"Best threshold on val set: warning_count >= {best_threshold} "
              f"(val F1={best_f1:.4f})")

        print("Running cppcheck on test set...")
        test_labels, test_counts = score_dataset(args.test_csv, tmpdir, cache)

    test_metrics = metrics_at_threshold(test_labels, test_counts, best_threshold)

    print(f"\n===== Cppcheck baseline test results (threshold={best_threshold}) =====")
    for k in ["acc", "precision", "recall", "f1", "fpr", "mcc"]:
        print(f"  {k}: {test_metrics[k]:.4f}")
    print(f"  (TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']})")

    with open(args.out, "w") as f:
        json.dump({"threshold": best_threshold, "val_f1": best_f1,
                    "test_metrics": test_metrics}, f, indent=2)
    print(f"\nSaved to {args.out}")

    # Per-sample test predictions, for a paired McNemar's test against
    # our own framework's predictions on the identical test set.
    test_preds = (test_counts >= best_threshold).astype(int)
    with open("cppcheck_per_sample.json", "w") as f:
        json.dump({"y_test": test_labels.tolist(), "cppcheck_pred": test_preds.tolist()}, f)
    print("Saved per-sample test predictions to cppcheck_per_sample.json")


if __name__ == "__main__":
    main()