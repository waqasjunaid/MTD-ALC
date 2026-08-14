#!/usr/bin/env python3
"""
check_truncation_root_cause.py

For each known-truncated sample (fragment starts mid-function, name is a
control-flow keyword), searches processed_data.csv's func_before column
for any row that CONTAINS the fragment as a substring but is longer --
i.e., a complete function that our extractor should have found but
didn't. If found: the extractor missed a real signature (fixable bug).
If not found anywhere: BigVul's own raw data is already truncated for
this row (inherited data-quality issue, not fixable in our pipeline).

Usage:
    python check_truncation_root_cause.py \
        --raw /path/to/processed_data.csv \
        --fragments fragments.json
"""
import argparse
import csv
import json
import sys


# The 8 fragments confirmed truncated, taken directly from
# bigvul_preprocessed.jsonl's func.code for these ids. Using a longer
# distinctive substring (not just the first 150 chars) to search more
# precisely -- pull enough unique tokens to avoid false-positive matches.
FRAGMENTS = {
    "99":   "cur->prev == NULL",
    "206":  "npasses_from_interlace_type",
    "216":  "parent_len += np->mb_len",
    "629":  "m_clusterPreloadCount",
    "691":  "zend_object_store_get_object(getThis()",
    "755":  "FT_THROW( Invalid_Argument )",
    "1369": "!bin->symtab || !bin->symstr",
    "1497": "hdl->recv_buf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="processed_data.csv")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)

    found = {}
    with open(args.raw, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            func_before = row.get("func_before", "") or ""
            for sid, needle in FRAGMENTS.items():
                if sid in found:
                    continue
                if needle in func_before:
                    found[sid] = {
                        "row_index": row.get("index"),
                        "cve_id": row.get("CVE ID"),
                        "func_before_len": len(func_before),
                        "func_before_head": func_before[:200],
                    }

    print(f"Searched processed_data.csv for {len(FRAGMENTS)} known fragments.\n")
    for sid in FRAGMENTS:
        if sid in found:
            info = found[sid]
            print(f"id={sid}: FOUND in raw data (row_index={info['row_index']}, "
                  f"CVE={info['cve_id']}, func_before length={info['func_before_len']} chars)")
            print(f"  raw func_before starts: {info['func_before_head']!r}")
        else:
            print(f"id={sid}: NOT FOUND anywhere in processed_data.csv's func_before column")
        print()

    n_found = len(found)
    print(f"\nSummary: {n_found}/{len(FRAGMENTS)} fragments found in a longer raw function.")
    if n_found == len(FRAGMENTS):
        print("-> All complete functions EXIST in the raw data. This is an "
              "extractor bug (fixable) -- the signature-detection regex "
              "is skipping past the real function start.")
    elif n_found == 0:
        print("-> None found. BigVul's own raw data appears to already be "
              "truncated for these rows (inherited limitation, not fixable "
              "in this pipeline).")
    else:
        print("-> Mixed result: some are extractor bugs, some are inherited "
              "truncation. Worth checking the NOT FOUND cases individually.")


if __name__ == "__main__":
    main()
