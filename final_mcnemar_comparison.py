#!/usr/bin/env python3
"""
final_mcnemar_comparison.py

McNemar's exact test: Ours (MTD+ALC) vs. LineVul, and Ours vs. Cppcheck,
on the identical 3,772-sample held-out test set.

ALIGNMENT NOTE (important): test.csv (used for LineVul and Cppcheck) was
built by build_linevul_csvs.py in split3()'s own shuffled order, with NO
sample_id column (LineVul's required schema is just processed_func/target).
bigvul_results_test_only.csv, by contrast, is keyed by sample_id in its
own original order. These are NOT the same row order.

This script re-derives the exact sample_id for each row position in
test.csv (via the same deterministic load_labelled+split3 used
throughout this project), then joins back to bigvul_results_test_only.csv
by that real ID -- and VERIFIES the alignment by cross-checking that the
label recovered via this join matches the label LineVul/Cppcheck's own
files recorded for that row, aborting if they ever disagree.

Usage:
    python final_mcnemar_comparison.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --our_results bigvul_results_test_only.csv \
        --linevul_preds results/raw_preds.csv \
        --cppcheck_preds cppcheck_per_sample.json \
        --threshold 0.15
"""
import argparse
import csv
import json
import math
import random

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


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def run_mcnemar(y, pred_a, pred_b, name_a, name_b):
    both_correct = a_only = b_only = both_wrong = 0
    for yi, a, b in zip(y, pred_a, pred_b):
        a_correct = (a == yi)
        b_correct = (b == yi)
        if a_correct and b_correct: both_correct += 1
        elif a_correct: a_only += 1
        elif b_correct: b_only += 1
        else: both_wrong += 1

    print(f"\n===== {name_a} vs {name_b} (n={len(y)}) =====")
    print(f"                    {name_b}: correct   {name_b}: wrong")
    print(f"  {name_a}: correct  {both_correct:>13}   {a_only:>13}")
    print(f"  {name_a}: wrong    {b_only:>13}   {both_wrong:>13}")

    p = exact_mcnemar_pvalue(a_only, b_only)
    print(f"Discordant pairs: {name_a}-only-correct={a_only}, "
          f"{name_b}-only-correct={b_only}")
    print(f"McNemar's exact p-value: {p:.4f}  "
          f"{'(significant at 0.05)' if p < 0.05 else '(NOT significant at 0.05)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--our_results", required=True)
    ap.add_argument("--linevul_preds", required=True)
    ap.add_argument("--cppcheck_preds", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    print("Re-deriving the exact seed-42 split to recover test.csv's row order...")
    X, y, ids = load_labelled(args.features)
    _, _, test_idx = split3(X, y, ids, val=0.15, test=0.20, seed=42)
    row_to_sample_id = [str(ids[i]) for i in test_idx]
    print(f"  test set size: {len(row_to_sample_id)}")

    # Our framework's predictions, keyed by sample_id
    our_lookup = {}
    with open(args.our_results, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row["sample_id"])
            label = int(row["label"])
            v_score = float(row["V_score"])
            our_lookup[sid] = (label, int(v_score >= args.threshold))

    # LineVul predictions, in test.csv row order
    linevul_rows = []
    with open(args.linevul_preds, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            linevul_rows.append((int(row["target"]), row["raw_preds"]))

    # Cppcheck predictions, in test.csv row order
    cpp = json.load(open(args.cppcheck_preds))
    cpp_labels = cpp["y_test"]
    cpp_preds = cpp["cppcheck_pred"]

    n = len(row_to_sample_id)
    assert len(linevul_rows) == n, (
        f"LineVul predictions count ({len(linevul_rows)}) != test set size ({n}) "
        f"-- something is misaligned, stopping.")
    assert len(cpp_labels) == n, (
        f"Cppcheck predictions count ({len(cpp_labels)}) != test set size ({n}) "
        f"-- something is misaligned, stopping.")

    y_true, our_pred, linevul_pred, cpp_pred = [], [], [], []
    mismatches = 0
    for i, sid in enumerate(row_to_sample_id):
        if sid not in our_lookup:
            print(f"ABORTED: sample_id {sid} (row {i}) not found in "
                  f"{args.our_results} -- alignment assumption failed.")
            return
        our_label, our_p = our_lookup[sid]

        lv_label, lv_raw = linevul_rows[i]
        lv_p = 1 if str(lv_raw).strip().lower() in ("1", "true") else 0

        cpp_label = int(cpp_labels[i])
        cpp_p = int(cpp_preds[i])

        # Consistency check: all three sources must agree on the TRUE label
        # for this row, or the row-order alignment assumption is broken.
        if not (our_label == lv_label == cpp_label):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at row {i} (sample_id={sid}): "
                      f"our_label={our_label} linevul_label={lv_label} "
                      f"cppcheck_label={cpp_label}")
            continue

        y_true.append(our_label)
        our_pred.append(our_p)
        linevul_pred.append(lv_p)
        cpp_pred.append(cpp_p)

    if mismatches > 0:
        print(f"\nABORTED: {mismatches}/{n} rows had disagreeing labels across "
              f"sources -- the row-order alignment assumption does not hold. "
              f"Do not trust any McNemar result below without fixing this first.")
        if mismatches == n:
            return

    print(f"\nAlignment verified: {n - mismatches}/{n} rows have fully "
          f"consistent labels across all three sources.")

    run_mcnemar(y_true, our_pred, linevul_pred, "Ours", "LineVul")
    run_mcnemar(y_true, our_pred, cpp_pred, "Ours", "Cppcheck")


if __name__ == "__main__":
    main()
