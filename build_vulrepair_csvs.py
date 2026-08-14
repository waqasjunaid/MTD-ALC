#!/usr/bin/env python3
"""
build_vulrepair_csvs.py

Builds train.csv/val.csv/test.csv for VulRepair (columns: source, target)
from the 548 samples already in results.jsonl (which already has
vulnerable_func and reference_fix -- no new content-matching needed).

Each of the 548 samples is bucketed into train/val/test using the SAME
seed-42 65/15/20 split used throughout this project, so the split is
consistent with everything else -- but note this results in a MUCH
smaller training set than VulRepair's original paper used (roughly
65% of 548 =~ 356 samples, vs. thousands in the original benchmark).
This is the deliberate reduced-scope tradeoff already agreed on.

Usage:
    python build_vulrepair_csvs.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --results results.jsonl \
        --outdir data/VulRepair/M1_VulRepair_PL-NL/data_ours
"""
import argparse
import csv
import json
import random
from pathlib import Path

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
    return mg(ptr, ntr), mg(pva, nva), mg(pte, nte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    print("Re-deriving the exact seed-42 split...")
    X, y, ids = load_labelled(args.features)
    train_idx, val_idx, test_idx = split3(X, y, ids, val=0.15, test=0.20, seed=42)
    train_ids = set(str(ids[i]) for i in train_idx)
    val_ids = set(str(ids[i]) for i in val_idx)
    test_ids = set(str(ids[i]) for i in test_idx)

    recs = [json.loads(l) for l in open(args.results) if l.strip()]
    print(f"Loaded {len(recs)} samples from {args.results}")

    buckets = {"train": [], "val": [], "test": []}
    unmatched = 0
    for r in recs:
        sid = str(r["id"])
        row = {"source": r["vulnerable_func"], "target": r["reference_fix"]}
        if sid in train_ids:
            buckets["train"].append(row)
        elif sid in val_ids:
            buckets["val"].append(row)
        elif sid in test_ids:
            buckets["test"].append(row)
        else:
            unmatched += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        path = outdir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "target"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {name}.csv: {len(rows)} samples -> {path}")

    if unmatched:
        print(f"\nWARNING: {unmatched} samples from {args.results} did not "
              f"match any split bucket -- investigate before trusting the "
              f"CSVs above.")

    n_train = len(buckets["train"])
    print(f"\n*** IMPORTANT: training set is only {n_train} samples ***")
    print(f"*** This must be disclosed as a reduced-scope limitation when   ***")
    print(f"*** reporting VulRepair results -- it is far smaller than the  ***")
    print(f"*** thousands of samples VulRepair's original paper trained on. ***")


if __name__ == "__main__":
    main()
