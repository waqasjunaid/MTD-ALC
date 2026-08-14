#!/usr/bin/env python3
"""
build_linevul_csvs.py

Builds train.csv / val.csv / test.csv for LineVul (columns: processed_func,
target) using the EXACT SAME seed-42 65/15/20 split as your MTD+ALC model
(reproducing load_labelled() + split3() from mtd/ml/train.py verbatim).

This guarantees LineVul and MTD+ALC are trained and evaluated on
IDENTICAL sample sets -- the actual "unified experiment" the reviewer
asked for, not just a fair-looking comparison.

Needs the raw function code, which mtd/ml/datasets/bigvul_features.jsonl
does NOT contain (only numeric features). Assumes bigvul_preprocessed.jsonl
(used elsewhere in your pipeline, e.g. regen_armA.py) has 'id', 'func'
(dict with 'code'), and 'label' for all 18,864 samples. Adjust
--preprocessed / the field-lookup logic below if your actual field names
differ.

Usage:
    python build_linevul_csvs.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --preprocessed /path/to/bigvul_preprocessed.jsonl \
        --outdir ../data/big-vul_dataset
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

FEATURE_DIM = 38


def load_labelled(path):
    """Verbatim reproduction of mtd/ml/train.py's load_labelled(), so the
    split is derived from the identical (X, y, ids) ordering used during
    actual MTD+ALC training."""
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
    """Verbatim reproduction of mtd/ml/train.py's split3()."""
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


def load_code_and_label(preprocessed_path):
    """sample_id -> (code, label) from bigvul_preprocessed.jsonl.
    ADJUST field names here if your schema differs."""
    lookup = {}
    with open(preprocessed_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("id"))
            func = rec.get("func", {})
            code = None
            if isinstance(func, dict):
                for k in ("code", "func_before", "source", "text", "body"):
                    if func.get(k):
                        code = str(func[k])
                        break
            elif isinstance(func, str) and func.strip():
                code = func
            label = rec.get("label")
            if code is not None and label is not None:
                lookup[sid] = (code, int(label))
    return lookup


def write_csv(path, ids, lookup, missing_log):
    n_written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["processed_func", "target"])
        for sid in ids:
            entry = lookup.get(str(sid))
            if entry is None:
                missing_log.append(sid)
                continue
            code, label = entry
            writer.writerow([code, label])
            n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="mtd/ml/datasets/bigvul_features.jsonl")
    ap.add_argument("--preprocessed", required=True, help="bigvul_preprocessed.jsonl (has raw code)")
    ap.add_argument("--outdir", required=True, help="e.g. ../data/big-vul_dataset")
    args = ap.parse_args()

    print("Re-deriving the exact seed-42 split used for MTD+ALC training...")
    X, y, ids = load_labelled(args.features)
    train_ids, val_ids, test_ids = split3(X, y, ids, val=0.15, test=0.20, seed=42)
    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    print("Loading raw function code and labels...")
    lookup = load_code_and_label(args.preprocessed)
    print(f"  loaded code+label for {len(lookup)} samples")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    missing = []
    n_train = write_csv(outdir / "train.csv", train_ids, lookup, missing)
    n_val = write_csv(outdir / "val.csv", val_ids, lookup, missing)
    n_test = write_csv(outdir / "test.csv", test_ids, lookup, missing)

    print(f"\nWrote: train.csv={n_train}  val.csv={n_val}  test.csv={n_test}")
    if missing:
        print(f"WARNING: {len(missing)} sample_ids had no code/label match "
              f"in --preprocessed and were skipped (first 10: {missing[:10]})")
        print("If this number is large, check the field names in "
              "load_code_and_label() against your actual bigvul_preprocessed.jsonl schema.")
    else:
        print("No missing samples -- every id in the split was matched successfully.")


if __name__ == "__main__":
    main()
