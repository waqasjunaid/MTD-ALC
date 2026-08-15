#!/usr/bin/env python3
"""
plot_roc_pr_curves.py

Regenerates Fig. 8 (ROC curve) and Fig. 9 (PR curve) using the CORRECTED
test-only results (bigvul_results_test_only.csv), replacing the earlier
versions which were computed on the leaked full-dataset file.

Usage:
    python plot_roc_pr_curves.py --results bigvul_results_test_only.csv \
        --out-roc fig_roc --out-pr fig_pr --theta-star 0.15
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def confusion_at(labels, scores, t):
    preds = (scores >= t).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    return tp, fp, tn, fn


def roc_curve(labels, scores):
    thresholds = np.concatenate([[1.0001], np.sort(np.unique(scores))[::-1], [-0.0001]])
    tprs, fprs = [], []
    for t in thresholds:
        tp, fp, tn, fn = confusion_at(labels, scores, t)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs)


def pr_curve(labels, scores):
    thresholds = np.concatenate([[1.0001], np.sort(np.unique(scores))[::-1], [-0.0001]])
    precisions, recalls = [], []
    for t in thresholds:
        tp, fp, tn, fn = confusion_at(labels, scores, t)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precisions.append(precision)
        recalls.append(recall)
    # sort by recall for a clean curve
    order = np.argsort(recalls)
    return np.array(recalls)[order], np.array(precisions)[order]


def auc_trapz(fprs, tprs):
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def average_precision(recalls, precisions):
    # standard AP: sum of precision * change in recall
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precisions):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results", required=True)
    ap_.add_argument("--out-roc", required=True)
    ap_.add_argument("--out-pr", required=True)
    ap_.add_argument("--theta-star", type=float, default=0.15)
    args = ap_.parse_args()

    labels, scores = [], []
    with open(args.results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["V_score"]))
    labels = np.array(labels)
    scores = np.array(scores)
    n_pos = int((labels == 1).sum())
    n_total = len(labels)
    prevalence = n_pos / n_total

    tp_s, fp_s, tn_s, fn_s = confusion_at(labels, scores, args.theta_star)
    tpr_star = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
    fpr_star = fp_s / (fp_s + tn_s) if (fp_s + tn_s) else 0.0
    prec_star = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 1.0

    # ---- ROC ----
    fprs, tprs = roc_curve(labels, scores)
    auc = auc_trapz(fprs, tprs)
    print(f"AUC = {auc:.4f}")
    print(f"Operating point: FPR={fpr_star:.4f}  TPR={tpr_star:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fprs, tprs, color="#1f6fb4", linewidth=2.5,
            label=f"MTD+ALC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
            label="Random baseline (AUC = 0.50)")
    ax.plot([fpr_star], [tpr_star], marker="o", markersize=11, color="#d62728",
            zorder=5, label=f"Operating point (FPR={fpr_star:.4f}, TPR={tpr_star:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve --- BigVul (held-out test set)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out_roc}.pdf")
    fig.savefig(f"{args.out_roc}.png", dpi=200)

    # ---- PR ----
    recalls, precisions = pr_curve(labels, scores)
    ap = average_precision(recalls, precisions)
    print(f"\nAP = {ap:.4f}")
    print(f"Operating point: P={prec_star:.4f}  R={tpr_star:.4f}")

    fig2, ax2 = plt.subplots(figsize=(6.5, 6))
    ax2.plot(recalls, precisions, color="#2ca02c", linewidth=2.5,
             label=f"MTD+ALC (AP = {ap:.4f})")
    ax2.axhline(prevalence, color="gray", linestyle="--", linewidth=1.5,
                label=f"Random baseline (AP = {prevalence:.4f})")
    ax2.plot([tpr_star], [prec_star], marker="o", markersize=11, color="#d62728",
             zorder=5, label=f"Operating point (P={prec_star:.4f}, R={tpr_star:.4f})")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Recall", fontsize=13)
    ax2.set_ylabel("Precision", fontsize=13)
    ax2.set_title("Precision-Recall Curve --- BigVul (held-out test set)", fontsize=14)
    ax2.legend(loc="lower left", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(f"{args.out_pr}.pdf")
    fig2.savefig(f"{args.out_pr}.png", dpi=200)

    print(f"\nSaved {args.out_roc}.pdf/.png and {args.out_pr}.pdf/.png")


if __name__ == "__main__":
    main()
