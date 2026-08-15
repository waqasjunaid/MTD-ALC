#!/usr/bin/env python3
"""
wilson_ci_boundary_metrics.py

Precision and FPR came out as degenerate [1.0000, 1.0000] / [0.0000, 0.0000]
bootstrap CIs on the test-only set because both are proportions with zero
observed events (0 false positives) -- a percentile bootstrap can't
manufacture an event that isn't in the data. The correct interval for a
zero-count (or any) proportion is the Wilson score interval.

Usage:
    python wilson_ci_boundary_metrics.py --results bigvul_results_test_only.csv --threshold 0.15
"""
import argparse
import csv
import math


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    labels, scores = [], []
    with open(args.results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["V_score"]))

    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= args.threshold else 0
        if pred == 1 and label == 1: tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 0: tn += 1
        else: fn += 1

    print(f"Confusion matrix (n={len(labels)}, threshold={args.threshold}):")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")

    prec_n = tp + fp
    prec_lo, prec_hi = wilson_ci(tp, prec_n)
    print(f"Precision = {tp}/{prec_n} = {tp/prec_n:.4f}")
    print(f"  Wilson 95% CI: [{prec_lo:.4f}, {prec_hi:.4f}]")

    fpr_n = fp + tn
    fpr_lo, fpr_hi = wilson_ci(fp, fpr_n)
    print(f"\nFPR = {fp}/{fpr_n} = {fp/fpr_n:.4f}")
    print(f"  Wilson 95% CI: [{fpr_lo:.4f}, {fpr_hi:.4f}]")

    rec_n = tp + fn
    rec_lo, rec_hi = wilson_ci(tp, rec_n)
    print(f"\n(For comparison) Recall = {tp}/{rec_n} = {tp/rec_n:.4f}")
    print(f"  Wilson 95% CI: [{rec_lo:.4f}, {rec_hi:.4f}]  "
          f"(vs. bootstrap CI reported earlier -- should be similar since "
          f"Recall isn't at a zero-count boundary)")


if __name__ == "__main__":
    main()
