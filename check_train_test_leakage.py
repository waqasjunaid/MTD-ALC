#!/usr/bin/env python3
"""
check_train_test_leakage.py

Table VI's numbers come from bigvul_results.csv. But mtd/ml/train.py's
split3() function shows the model was trained on only 65% of BigVul
(a seeded 65/15/20 train/val/test split) -- while your paper's text says
Table VI is "reported on the full dataset of 18,864 samples." This
script settles whether that's actually a problem.

Since the split uses a fixed seed (42), it's fully deterministic and can
be re-derived exactly as it was during training, without needing any
file that saved the original split (which wasn't persisted -- split3's
callers discard the id lists). This script reproduces load_labelled()
and split3() verbatim from mtd/ml/train.py, then checks what fraction of
bigvul_results.csv's rows fall inside the re-derived test set vs. the
train/val portion the model was actually fit on.

Usage:
    python check_train_test_leakage.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --results bigvul_results.csv
"""
import argparse
import csv
import json
import random
import sys

import numpy as np

FEATURE_DIM = 38  # from mtd/ml/feature_extractor.py


def load_labelled(path):
    X, y, ids = [], [], []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label", -1) == -1:
                skipped += 1
                continue
            feats = rec.get("features", [])
            if len(feats) != FEATURE_DIM:
                continue
            X.append(feats)
            y.append(int(rec["label"]))
            ids.append(rec["sample_id"])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"loaded from features file: labelled={len(y)}  skipped_unlabelled={skipped}")
    return X, y, ids


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
    tr = mg(ptr, ntr)
    va = mg(pva, nva)
    te = mg(pte, nte)
    return ([ids[i] for i in tr], [ids[i] for i in va], [ids[i] for i in te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="mtd/ml/datasets/bigvul_features.jsonl")
    ap.add_argument("--results", required=True, help="bigvul_results.csv")
    args = ap.parse_args()

    X, y, ids = load_labelled(args.features)
    train_ids, val_ids, test_ids = split3(X, y, ids, val=0.15, test=0.20, seed=42)

    train_ids_set = set(str(i) for i in train_ids)
    val_ids_set = set(str(i) for i in val_ids)
    test_ids_set = set(str(i) for i in test_ids)

    print(f"\nre-derived split sizes: train={len(train_ids_set)}  "
          f"val={len(val_ids_set)}  test={len(test_ids_set)}")

    results_ids = []
    with open(args.results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results_ids.append(str(row["sample_id"]))

    n = len(results_ids)
    n_in_train = sum(1 for i in results_ids if i in train_ids_set)
    n_in_val = sum(1 for i in results_ids if i in val_ids_set)
    n_in_test = sum(1 for i in results_ids if i in test_ids_set)
    n_unknown = n - n_in_train - n_in_val - n_in_test

    print(f"\nbigvul_results.csv has {n} rows. Of these:")
    print(f"  in TRAIN split (model was fit on these):  {n_in_train} ({100*n_in_train/n:.1f}%)")
    print(f"  in VAL split (used for threshold/tuning):  {n_in_val} ({100*n_in_val/n:.1f}%)")
    print(f"  in TEST split (genuinely held out):        {n_in_test} ({100*n_in_test/n:.1f}%)")
    print(f"  not found in any split (e.g. unlabelled):  {n_unknown} ({100*n_unknown/n:.1f}%)")

    print()
    if n_in_train + n_in_val > 0.05 * n:
        print("*** LEAKAGE CONFIRMED: a substantial fraction of bigvul_results.csv's ***")
        print("*** rows were in the model's own train/val split. Table VI's numbers ***")
        print("*** as currently computed include performance on data the model was  ***")
        print("*** fit on, and are not a valid held-out evaluation.                 ***")
    else:
        print("No meaningful leakage detected -- bigvul_results.csv appears to be "
              "(at least very close to) the genuine held-out test set.")


if __name__ == "__main__":
    main()
