#!/usr/bin/env python3
"""
filter_to_test_only.py

Filters bigvul_results.csv down to ONLY the genuine held-out test split
(re-derived deterministically from the seed-42 split3() used in
mtd/ml/train.py), producing a corrected file to recompute Table VI and
Figures 8-10 without train/val leakage.

Usage:
    python filter_to_test_only.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --results bigvul_results.csv \
        --out bigvul_results_test_only.csv
"""
import argparse
import csv
import json
import random
import sys

import numpy as np

FEATURE_DIM = 38


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
    return (mg(ptr, ntr), mg(pva, nva), mg(pte, nte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, y, ids = load_labelled(args.features)
    tr_idx, va_idx, te_idx = split3(X, y, ids, val=0.15, test=0.20, seed=42)
    test_ids = set(str(ids[i]) for i in te_idx)

    n_written = 0
    n_total = 0
    with open(args.results, newline="", encoding="utf-8") as fin, \
         open(args.out, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            n_total += 1
            if str(row["sample_id"]) in test_ids:
                writer.writerow(row)
                n_written += 1

    print(f"read {n_total} rows, wrote {n_written} test-only rows to {args.out}")
    print(f"(expected ~{len(test_ids)} based on the re-derived split)")


if __name__ == "__main__":
    main()
