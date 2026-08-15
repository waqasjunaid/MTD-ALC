#!/usr/bin/env python3
"""
task_ablation.py

Answers the reviewer comment "the paper lacks ablation studies to
determine the contribution of the individual components": retrains the
logistic regression classifier on cumulative subsets of the four MTD
tasks' features, using the EXACT same StandardScaler/LogisticRegression
implementation, hyperparameters, and seed-42 65/15/20 split as the real
trained model -- so these numbers are directly comparable to Table VI.

Feature-to-task mapping (from mtd/ml/feature_extractor.py FEATURE_NAMES,
38 dims total):
  indices  0- 4 : Task 1 (Block A)
  indices  5- 9 : Task 2 (Block B)
  indices 10-15 : Task 3 (Block C)
  indices 16-23 : Task 4 (Block D)
  indices 24-29 : shared metadata (pre-task, from line_map -- always kept)
  index      30 : Task 4-derived (loop_count from control_flow) -- kept
                  only once Task 4 is included
  index      31 : f0 = Task1 x Task3 interaction
  index      32 : f1_ = Task1 x Task4 interaction
  index      33 : f2_ = Task3 x Task4 interaction
  index      34 : f3 = Task1-internal interaction (a1 x a2) -- kept once
                  Task 1 is included
  index      35 : f4 = Task4-internal interaction (d4 x d5) -- kept once
                  Task 4 is included
  index      36 : f5 = Task2 x Task4 interaction
  index      37 : f6 = Task3 x Task4 interaction

For each configuration, features whose required task(s) are not yet
included are zeroed out (not removed from the vector -- this keeps the
scaler/model architecture identical across configs, isolating the
effect of information content rather than dimensionality).

Usage:
    python task_ablation.py --features mtd/ml/datasets/bigvul_features.jsonl
"""
import argparse
import json
import math
import random

import numpy as np

FEATURE_DIM = 38


# ---------------------------------------------------------------------
# Exact copies of the real training pipeline's classes (from
# mtd/ml/train.py), so this ablation uses IDENTICAL fitting behavior.
# ---------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegression:
    def __init__(self, lr=0.05, epochs=300, batch_size=64, l2=1e-4, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.bs = batch_size
        self.l2 = l2
        self.seed = seed
        self.w = None
        self.b = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.w = rng.randn(d).astype(np.float32) * 0.01
        self.b = np.float32(0.0)
        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]
            for s in range(0, n, self.bs):
                Xb = Xs[s:s + self.bs]
                yb = ys[s:s + self.bs].astype(np.float32)
                p = _sig(Xb @ self.w + self.b)
                e = p - yb
                self.w -= self.lr * (Xb.T @ e / len(yb) + self.l2 * self.w)
                self.b -= self.lr * e.mean()
        return self

    def predict_proba(self, X):
        return _sig(X @ self.w + self.b)


def find_opt_threshold(probs, y):
    best = {"threshold": 0.50, "f1": 0.0}
    for t in [i / 100 for i in range(5, 96)]:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        if f1 > best["f1"]:
            best = {"threshold": round(t, 2), "f1": round(f1, 4)}
    return best["threshold"]


# ---------------------------------------------------------------------
# Data loading / splitting (verbatim reproduction, as used elsewhere
# in this project's verification scripts)
# ---------------------------------------------------------------------
def load_labelled(path):
    X, y, ids = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label", -1) == -1:
                continue
            feats = rec.get("features", [])
            if len(feats) != FEATURE_DIM:
                continue
            X.append(feats)
            y.append(int(rec["label"]))
            ids.append(rec["sample_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), ids


def split3(X, y, ids, val=0.15, test=0.20, seed=42):
    rng = random.Random(seed)
    pos = [i for i, l in enumerate(y) if l == 1]
    neg = [i for i, l in enumerate(y) if l == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(lst):
        n1 = max(1, int(len(lst) * test))
        n2 = max(1, int(len(lst) * val))
        return lst[:n1], lst[n1:n1 + n2], lst[n1 + n2:]

    def mg(a, b):
        idx = a + b
        rng.shuffle(idx)
        return idx

    pte, pva, ptr = cut(pos)
    nte, nva, ntr = cut(neg)
    return mg(ptr, ntr), mg(pva, nva), mg(pte, nte)


# ---------------------------------------------------------------------
# Task-masking logic
# ---------------------------------------------------------------------
TASK_BLOCKS = {
    1: list(range(0, 5)),
    2: list(range(5, 10)),
    3: list(range(10, 16)),
    4: list(range(16, 24)),
}
SHARED_METADATA = list(range(24, 30))          # e0-e5, always kept
TASK4_METADATA = [30]                          # e6, needs Task 4

# interaction index -> set of tasks required (besides always-available T1-internal/T4-internal cases)
INTERACTIONS = {
    31: {1, 3},   # f0  = T1 x T3
    32: {1, 4},   # f1_ = T1 x T4
    33: {3, 4},   # f2_ = T3 x T4
    34: {1},      # f3  = T1-internal
    35: {4},      # f4  = T4-internal
    36: {2, 4},   # f5  = T2 x T4
    37: {3, 4},   # f6  = T3 x T4
}


def mask_features(X, included_tasks: set):
    """Zero out any feature whose required task(s) are not fully included."""
    Xm = X.copy()
    for task_id, idxs in TASK_BLOCKS.items():
        if task_id not in included_tasks:
            Xm[:, idxs] = 0.0
    if 4 not in included_tasks:
        Xm[:, TASK4_METADATA] = 0.0
    for idx, required in INTERACTIONS.items():
        if not required.issubset(included_tasks):
            Xm[:, idx] = 0.0
    return Xm


def compute_metrics(labels, scores, threshold):
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
    mcc_denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(pos_scores) and len(neg_scores):
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos_ranks = ranks[labels == 1]
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        auc = (float(np.sum(pos_ranks)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    else:
        auc = float("nan")

    return {"acc": acc, "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "auc": auc, "mcc": mcc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    args = ap.parse_args()

    print("Loading features and re-deriving the exact seed-42 split...")
    X, y, ids = load_labelled(args.features)
    train_idx, val_idx, test_idx = split3(X, y, ids, val=0.15, test=0.20, seed=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}\n")

    configs = [
        ("Task 1 only", {1}),
        ("Task 1+2", {1, 2}),
        ("Task 1+2+3", {1, 2, 3}),
        ("Task 1+2+3+4 (full)", {1, 2, 3, 4}),
    ]

    print(f"{'Config':<22} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'FPR':>7} {'AUC':>7} {'MCC':>7}")
    print("-" * 80)

    results = {}
    for name, tasks in configs:
        Xtr_m = mask_features(X_train, tasks)
        Xva_m = mask_features(X_val, tasks)
        Xte_m = mask_features(X_test, tasks)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_m)
        Xva_s = scaler.transform(Xva_m)
        Xte_s = scaler.transform(Xte_m)

        model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4, seed=42)
        model.fit(Xtr_s, y_train)

        val_probs = model.predict_proba(Xva_s)
        threshold = find_opt_threshold(val_probs, y_val)

        test_probs = model.predict_proba(Xte_s)
        m = compute_metrics(y_test, test_probs, threshold)
        results[name] = {"threshold": threshold, **m}

        print(f"{name:<22} {m['acc']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['fpr']:>7.4f} "
              f"{m['auc']:>7.4f} {m['mcc']:>7.4f}")

    with open("task_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to task_ablation_results.json")

    # Also save per-sample test predictions for Task-1-only and the full
    # model, so a paired significance test (McNemar's) can be run on the
    # two meaningfully-different configurations.
    per_sample = {"y_test": y_test.tolist()}
    for name, tasks in [("task1_only", {1}), ("full", {1, 2, 3, 4})]:
        Xtr_m = mask_features(X_train, tasks)
        Xte_m = mask_features(X_test, tasks)
        Xva_m = mask_features(X_val, tasks)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr_m)
        Xva_s = scaler.transform(Xva_m)
        Xte_s = scaler.transform(Xte_m)
        model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4, seed=42)
        model.fit(Xtr_s, y_train)
        threshold = find_opt_threshold(model.predict_proba(Xva_s), y_val)
        test_probs = model.predict_proba(Xte_s)
        preds = (test_probs >= threshold).astype(int)
        per_sample[name] = preds.tolist()

    with open("task_ablation_per_sample.json", "w") as f:
        json.dump(per_sample, f)
    print("Saved per-sample predictions to task_ablation_per_sample.json")


if __name__ == "__main__":
    main()