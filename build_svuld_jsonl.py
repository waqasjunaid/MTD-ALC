#!/usr/bin/env python3
"""
build_svuld_jsonl.py

Builds SVulD's train.jsonl / valid.jsonl / test.jsonl using the EXACT
SAME seed-42 65/15/20 split as MTD+ALC and the LineVul retraining run.

Per sample, writes: {"index": <int>, "code": <str>, "contrast": <str>,
"label": <int>}.

NOTE on 'contrast': SVulD's official reported results (per their
README) use --r_drop, whose loss (see model.py's forward()) only
re-embeds `code` twice for consistency regularization -- it never
uses `contrast` at all. TextDataset still requires the field to exist
and be tokenizable, so we set contrast=code as a faithful placeholder
matching their actual --r_drop training path, rather than building
real reference-fix pairs (only used by the separate, non-headline
--simct flag) that would require re-running the fragile
processed_data.csv content-matching at a larger scale for no benefit.

Usage:
    python build_svuld_jsonl.py \
        --features mtd/ml/datasets/bigvul_features.jsonl \
        --preprocessed data/bigvul_preprocessed.jsonl \
        --outdir data/SVulD-master/SVulD-master/Source_Code/SVulD/dataset_ours
"""
import argparse
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
    return (mg(ptr, ntr), mg(pva, nva), mg(pte, nte))


def load_code_and_label(preprocessed_path):
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


def write_jsonl(path, ids, lookup, missing_log):
    n_written = 0
    with open(path, "w", encoding="utf-8") as f:
        for sid in ids:
            entry = lookup.get(str(sid))
            if entry is None:
                missing_log.append(sid)
                continue
            code, label = entry
            try:
                index = int(sid)
            except ValueError:
                index = n_written  # fallback if sample_id isn't purely numeric
            rec = {
                "index": index,
                "code": code,
                "contrast": code,  # placeholder -- see module docstring
                "label": label,
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    print("Re-deriving the exact seed-42 split...")
    X, y, ids = load_labelled(args.features)
    train_ids, val_ids, test_ids = split3(X, y, ids, val=0.15, test=0.20, seed=42)
    print(f"  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    print("Loading code + label...")
    lookup = load_code_and_label(args.preprocessed)
    print(f"  loaded {len(lookup)} samples")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    missing = []
    n_train = write_jsonl(outdir / "train.jsonl", train_ids, lookup, missing)
    n_val = write_jsonl(outdir / "valid.jsonl", val_ids, lookup, missing)
    n_test = write_jsonl(outdir / "test.jsonl", test_ids, lookup, missing)

    print(f"\nWrote: train.jsonl={n_train}  valid.jsonl={n_val}  test.jsonl={n_test}")
    if missing:
        print(f"WARNING: {len(missing)} missing (first 10: {missing[:10]})")
    else:
        print("No missing samples.")


if __name__ == "__main__":
    main()
