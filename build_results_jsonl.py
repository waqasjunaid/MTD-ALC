#!/usr/bin/env python3
"""
build_results_jsonl.py

Joins diagnosis_records.jsonl (which has sample_id, vulnerable_func,
reference_fix, has_reference_fix, dataset) with the regenerated Arm A
patches in patches_armA/{sample_id}.c, producing results.jsonl for
repair_ablation_metrics.py.

This is Arm A (diagnosis-guided) / correctness-only: no Arm B / no
ablation table, matching the "two axes: coverage + correctness" framing
in your paper's Repair section.

Usage:
    python build_results_jsonl.py \
        --records diagnosis_records.jsonl \
        --patches patches_armA \
        --out results.jsonl

Skips (and reports) any sample_id that:
  - is missing a patch file in --patches (e.g. one of the 3 "skipped"
    from armA_run.log, or filtered out upstream)
  - has has_reference_fix == false (no ground truth to score against)
"""

import argparse
import json
import os
import re

_HEADER_RE = re.compile(
    r"^\s*/\*.*?Repaired by LLM Repair Module.*?\*/\s*",
    re.DOTALL,
)


def strip_archive_header(text: str) -> str:
    """Remove the standardized metadata header regen_armA.py stamps onto
    every archived patch file, so scoring compares code to code."""
    m = _HEADER_RE.match(text)
    if m:
        return text[m.end():]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="diagnosis_records.jsonl")
    ap.add_argument("--patches", required=True, help="patches_armA directory")
    ap.add_argument("--out", required=True, help="output results.jsonl")
    args = ap.parse_args()

    n_total = 0
    n_no_patch = 0
    n_no_ref = 0
    n_written = 0
    missing_ids = []

    with open(args.records) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            sample_id = str(rec.get("id"))
            dataset = rec.get("dataset", "bigvul")

            ref_fix = rec.get("reference_fix", "")
            vuln = rec.get("vulnerable_func", "")
            if not ref_fix or not ref_fix.strip() or ref_fix.strip() == vuln.strip():
                n_no_ref += 1
                continue

            patch_path = os.path.join(args.patches, f"{sample_id}.c")
            if not os.path.exists(patch_path):
                n_no_patch += 1
                missing_ids.append(sample_id)
                continue

            with open(patch_path, encoding="utf-8", errors="replace") as pf:
                generated_patch = strip_archive_header(pf.read())

            out_rec = {
                "id": sample_id,
                "dataset": dataset,
                "arm": "diag",
                "generated_patch": generated_patch,
                "reference_fix": rec["reference_fix"],
                "vulnerable_func": rec["vulnerable_func"],
            }
            fout.write(json.dumps(out_rec) + "\n")
            n_written += 1

    print(f"records read:        {n_total}")
    print(f"skipped (no ref fix): {n_no_ref}")
    print(f"skipped (no patch):   {n_no_patch}")
    print(f"written to {args.out}: {n_written}")
    if missing_ids:
        print(f"\nsample_ids with no patch file (first 20 shown):")
        print(missing_ids[:20])


if __name__ == "__main__":
    main()