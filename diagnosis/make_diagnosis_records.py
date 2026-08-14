# -*- coding: utf-8 -*-
"""
make_diagnosis_records.py

Builds diagnosis_records.jsonl for the repair ablation.

Join logic (robust, id-based):
  1. results CSV  (bigvul_results.csv)  -- keyed by sample_id
        gives: label, alc_decision, dominant_cwe, error_source, outlier_task
  2. preprocessed JSONL  (bigvul_preprocessed.jsonl)  -- keyed by id == sample_id
        gives: the vulnerable function body that the pipeline actually analyzed
  3. raw dataset CSV  (processed_data.csv)  -- OPTIONAL
        used ONLY to retrieve the reference fix (func_after) by matching the
        vulnerable function text (func_before) exactly.

Keeps only the truly-vulnerable flagged subset:
      label == 1  AND  alc_decision == 'untrustworthy'

Usage:
  python make_diagnosis_records.py \
      --results bigvul_results.csv \
      --jsonl   /path/bigvul_preprocessed.jsonl \
      --raw     /path/processed_data.csv \
      --dataset bigvul \
      --out     diagnosis_records.jsonl

If the JSONL already contains the fix, --raw can be omitted.
"""

import argparse
import json
import pandas as pd


JSONL_VULN_KEYS = ["func_before", "vulnerable_func", "code", "source"]
JSONL_FIX_KEYS  = ["func_after", "fixed_func", "reference_fix", "fix"]
RAW_VULN_COL    = "func_before"
RAW_FIX_COL     = "func_after"


def norm(s):
    if s is None:
        return ""
    return " ".join(str(s).split())


def extract_func_text(func_field):
    if isinstance(func_field, str):
        return func_field
    if isinstance(func_field, dict):
        for k in ("code", "func_before", "source", "text", "body"):
            if k in func_field and func_field[k]:
                return func_field[k]
    return ""


def load_jsonl(path):
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = str(r.get("id", i))
            by_id[rid] = r
    return by_id


def get_vuln_from_jsonl(rec):
    for k in JSONL_VULN_KEYS:
        if k in rec and rec[k]:
            return str(rec[k])
    return extract_func_text(rec.get("func", {}))


def get_fix_from_jsonl(rec):
    for k in JSONL_FIX_KEYS:
        if k in rec and rec[k]:
            return str(rec[k])
    return None


def build_raw_fix_index(raw_path):
    raw = pd.read_csv(raw_path, low_memory=False)
    if RAW_VULN_COL not in raw.columns or RAW_FIX_COL not in raw.columns:
        raise SystemExit(
            "Raw file missing '%s' or '%s'. Columns present: %s"
            % (RAW_VULN_COL, RAW_FIX_COL, list(raw.columns))
        )
    idx = {}
    for _, r in raw.iterrows():
        vb = r.get(RAW_VULN_COL)
        fa = r.get(RAW_FIX_COL)
        if pd.isna(vb) or pd.isna(fa):
            continue
        idx[norm(vb)] = str(fa)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    res = pd.read_csv(args.results)
    jsonl = load_jsonl(args.jsonl)

    subset = res[(res["label"] == 1) &
                 (res["alc_decision"].astype(str).str.lower() == "untrustworthy")].copy()
    print("%s: %d truly-vulnerable flagged samples (from %d rows)"
          % (args.dataset, len(subset), len(res)))

    raw_idx = None
    if args.raw:
        print("Building reference-fix index from raw dataset (may take a minute)...")
        raw_idx = build_raw_fix_index(args.raw)
        print("  indexed %d func_before -> func_after pairs" % len(raw_idx))

    mode = "a" if args.append else "w"
    written = 0
    no_jsonl = 0
    no_fix = 0

    with open(args.out, mode, encoding="utf-8") as out:
        for _, row in subset.iterrows():
            sid = str(row["sample_id"])
            rec = jsonl.get(sid)
            if rec is None:
                no_jsonl += 1
                continue

            vuln = get_vuln_from_jsonl(rec)
            if not vuln:
                no_jsonl += 1
                continue

            fix = get_fix_from_jsonl(rec)
            if fix is None and raw_idx is not None:
                fix = raw_idx.get(norm(vuln))
            if not fix:
                no_fix += 1
                continue

            out.write(json.dumps({
                "id": sid,
                "dataset": args.dataset,
                "vulnerable_func": vuln,
                "reference_fix": fix,
                "dominant_cwe": str(row.get("dominant_cwe", "")),
                "error_source": str(row.get("error_source", "")),
                "outlier_task": str(row.get("outlier_task", "")),
            }) + "\n")
            written += 1

    print("  wrote %d records to %s" % (written, args.out))
    if no_jsonl:
        print("  %d samples had no JSONL match on sample_id/id" % no_jsonl)
    if no_fix:
        print("  %d samples had no reference fix "
              "(vulnerable function not found in raw func_before column)" % no_fix)


if __name__ == "__main__":
    main()