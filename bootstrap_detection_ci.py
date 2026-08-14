#!/usr/bin/env python3
"""
bootstrap_detection_ci.py

Computes 95% bootstrap confidence intervals for Table VI's detection
metrics (Accuracy, Precision, Recall, F1, FPR, AUC, MCC) using the real
per-sample labels and V_scores in bigvul_results.csv.

This answers the reviewer comment about missing statistical rigor for
YOUR OWN reported numbers. It does NOT (and cannot) provide a
significance test against the 5 baselines (Devign, DeepDFA, IVDetect,
SVulD, LineVul), since those are point estimates taken from their
original papers -- no per-sample predictions are available to compare
against. That limitation should be stated explicitly in the paper
rather than worked around.

Usage:
    python bootstrap_detection_ci.py \
        --results bigvul_results.csv \
        --threshold 0.15 \
        --n-boot 2000
"""
import argparse
import csv
import random
import sys

import numpy as np


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0

    # AUC via the Mann-Whitney U / rank-sum equivalence -- no sklearn dependency.
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        auc = float("nan")
    else:
        all_scores = np.concatenate([pos_scores, neg_scores])
        ranks = np.argsort(np.argsort(all_scores)) + 1  # average-tie-free ranks (fine for bootstrap CI purposes)
        rank_sum_pos = ranks[: len(pos_scores)].sum()
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return {"acc": acc, "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "auc": auc, "mcc": mcc}


def bootstrap_ci(labels, scores, threshold, n_boot, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    metric_names = ["acc", "precision", "recall", "f1", "fpr", "auc", "mcc"]
    samples = {m: [] for m in metric_names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(labels[idx], scores[idx], threshold)
        for k in metric_names:
            samples[k].append(m[k])

    results = {}
    for k in metric_names:
        arr = np.array(samples[k])
        arr = arr[~np.isnan(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5])
        results[k] = (lo, hi)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="bigvul_results.csv")
    ap.add_argument("--threshold", type=float, default=None,
                     help="if omitted, checks both 0.15 and 0.225 against your "
                          "reported Table VI numbers first to resolve which was "
                          "actually used")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    labels, scores = [], []
    with open(args.results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["V_score"]))

    labels = np.array(labels)
    scores = np.array(scores)

    if args.threshold is None:
        print("No --threshold given -- checking which candidate reproduces your "
              "reported Table VI numbers (Acc=0.9857, P=0.9975, R=0.7469, "
              "F1=0.8542, FPR=0.0001, MCC=0.8567):\n")
        for cand in (0.15, 0.225):
            m = compute_metrics(labels, scores, cand)
            print(f"threshold={cand}:  Acc={m['acc']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  FPR={m['fpr']:.4f}  "
                  f"MCC={m['mcc']:.4f}")
        print("\nRe-run with --threshold <the one that matches> to compute "
              "the bootstrap CI at the correct operating point. If NEITHER "
              "matches closely, stop and let's investigate further before "
              "reporting any CI.")
        return

    point = compute_metrics(labels, scores, args.threshold)
    ci = bootstrap_ci(labels, scores, args.threshold, args.n_boot)

    print(f"n = {len(labels)}   threshold = {args.threshold}   "
          f"bootstrap resamples = {args.n_boot}\n")
    print(f"{'Metric':<12} {'Point est.':>12} {'95% CI':>22}")
    print("-" * 48)
    for k in ["acc", "precision", "recall", "f1", "fpr", "auc", "mcc"]:
        lo, hi = ci[k]
        print(f"{k:<12} {point[k]:>12.4f}   [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
