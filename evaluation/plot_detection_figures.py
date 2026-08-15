#!/usr/bin/env python3
"""
plot_detection_figures.py

Regenerates Figures 8 (ROC), 9 (Precision-Recall), and 10 (Precision/
Recall/F1 vs Threshold) from the CORRECTED, test-only results file
(bigvul_results_test_only.csv), replacing versions that were computed
on the full train+val+test set (data leakage).

Usage:
    python plot_detection_figures.py \
        --results bigvul_results_test_only.csv \
        --threshold 0.15 \
        --outdir .
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    labels, scores = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["V_score"]))
    return np.array(labels), np.array(scores)


def compute_pr_f1(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return precision, recall, f1, fpr


def roc_curve(labels, scores):
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    P = labels.sum()
    N = len(labels) - P
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    tpr = np.concatenate([[0], tps / P, [1]])
    fpr = np.concatenate([[0], fps / N, [1]])
    # AUC via trapezoidal rule on the step curve (manual implementation for
    # NumPy version compatibility -- np.trapz was removed in NumPy 2.0+)
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def pr_curve_and_ap(labels, scores):
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    P = labels.sum()
    tps = np.cumsum(labels_sorted == 1)
    fps = np.cumsum(labels_sorted == 0)
    precision = tps / (tps + fps)
    recall = tps / P
    # standard step-function Average Precision (matches sklearn's definition)
    recall_padded = np.concatenate([[0], recall])
    precision_padded = np.concatenate([[precision[0]], precision])
    ap = np.sum((recall_padded[1:] - recall_padded[:-1]) * precision_padded[1:])
    return recall, precision, ap


def plot_roc(labels, scores, threshold, outdir):
    fpr_curve, tpr_curve, auc = roc_curve(labels, scores)
    _, _, _, op_fpr = compute_pr_f1(labels, scores, threshold)
    op_tpr = compute_pr_f1(labels, scores, threshold)[1]  # recall == TPR

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(fpr_curve, tpr_curve, color="#1f6fb4", linewidth=2.5,
            label=f"MTD+ALC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
            label="Random baseline (AUC = 0.50)")
    ax.plot([op_fpr], [op_tpr], marker="o", markersize=11, color="#d62728",
            zorder=5, label=f"Operating point (FPR={op_fpr:.4f}, TPR={op_tpr:.4f})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=13)
    ax.set_title("ROC Curve — BigVul (held-out test set)", fontsize=15)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_roc.pdf")
    fig.savefig(f"{outdir}/fig_roc.png", dpi=200)
    print(f"ROC: AUC={auc:.4f}  operating point FPR={op_fpr:.4f} TPR={op_tpr:.4f}")


def plot_pr(labels, scores, threshold, outdir):
    recall_curve, precision_curve, ap = pr_curve_and_ap(labels, scores)
    op_prec, op_rec, _, _ = compute_pr_f1(labels, scores, threshold)
    base_rate = labels.sum() / len(labels)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(recall_curve, precision_curve, color="#2ca02c", linewidth=2.5,
            label=f"MTD+ALC (AP = {ap:.4f})")
    ax.plot([0, 1], [base_rate, base_rate], color="gray", linestyle="--",
            linewidth=1.5, label=f"Random baseline (AP = {base_rate:.4f})")
    ax.plot([op_rec], [op_prec], marker="o", markersize=11, color="#d62728",
            zorder=5, label=f"Operating point (P={op_prec:.4f}, R={op_rec:.4f})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve — BigVul (held-out test set)", fontsize=15)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_pr.pdf")
    fig.savefig(f"{outdir}/fig_pr.png", dpi=200)
    print(f"PR: AP={ap:.4f}  operating point P={op_prec:.4f} R={op_rec:.4f}")


def plot_threshold(labels, scores, theta_star, outdir):
    thresholds = np.linspace(0.01, 0.99, 400)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        p, r, f1, _ = compute_pr_f1(labels, scores, t)
        precisions.append(p); recalls.append(r); f1s.append(f1)
    p_star, r_star, f1_star, _ = compute_pr_f1(labels, scores, theta_star)

    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.plot(thresholds, f1s, color="#1f6fb4", linewidth=2.5, label="F1 Score")
    ax.plot(thresholds, precisions, color="#2ca02c", linestyle="--",
            linewidth=2.2, label="Precision")
    ax.plot(thresholds, recalls, color="#ff7f0e", linestyle=":",
            linewidth=2.4, label="Recall")
    ax.axvline(theta_star, color="black", linestyle="--", linewidth=2.2,
               label=rf"Optimal $\theta^*$ = {theta_star:g}")
    ax.plot([theta_star], [f1_star], marker="o", markersize=11,
            color="#d62728", zorder=5)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel(r"Classification Threshold $\theta$", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Precision, Recall & F1 vs Threshold — BigVul (held-out test set)", fontsize=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{outdir}/fig_threshold.pdf")
    fig.savefig(f"{outdir}/fig_threshold.png", dpi=200)
    print(f"Threshold: at theta*={theta_star}: P={p_star:.4f} R={r_star:.4f} F1={f1_star:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    labels, scores = load(args.results)
    print(f"n={len(labels)}  positives={labels.sum()}  negatives={len(labels)-labels.sum()}\n")

    plot_roc(labels, scores, args.threshold, args.outdir)
    plot_pr(labels, scores, args.threshold, args.outdir)
    plot_threshold(labels, scores, args.threshold, args.outdir)
    print("\nSaved: fig_roc.{pdf,png}  fig_pr.{pdf,png}  fig_threshold.{pdf,png}")


if __name__ == "__main__":
    main()
