#!/usr/bin/env python3
"""
check_reference_fix_quality.py

Sanity-checks diagnosis_records.jsonl: for each record, does
reference_fix actually look like a fixed version of the SAME function
as vulnerable_func, or does it look like it belongs to a different
function entirely (a join/match bug)?

Heuristic: extract the function name (the identifier right before the
first '(' on the first non-blank line) from both vulnerable_func and
reference_fix. If they don't match, and the name doesn't appear
anywhere else in reference_fix either, flag it as SUSPECT.

Usage:
    python check_reference_fix_quality.py diagnosis_records.jsonl
"""

import json
import re
import sys


def extract_func_name(code: str):
    """Best-effort: grab the identifier immediately before the first '('
    that looks like a function definition (skips common control keywords)."""
    if not code:
        return None
    # collapse whitespace to make this robust to newlines/indentation
    flat = re.sub(r"\s+", " ", code)
    # find identifier immediately followed by '(' -- take the first
    # candidate that isn't a control-flow keyword
    skip = {"if", "for", "while", "switch", "return", "sizeof", "catch"}
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", flat):
        name = m.group(1)
        if name not in skip:
            return name
    return None


def main(path):
    n = 0
    n_name_match = 0
    n_name_present_elsewhere = 0
    n_suspect = 0
    suspects = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            vuln_name = extract_func_name(r.get("vulnerable_func", ""))
            fix_name = extract_func_name(r.get("reference_fix", ""))

            if vuln_name and fix_name and vuln_name == fix_name:
                n_name_match += 1
            elif vuln_name and vuln_name in (r.get("reference_fix") or ""):
                n_name_present_elsewhere += 1
            else:
                n_suspect += 1
                suspects.append({
                    "id": r.get("id"),
                    "vuln_name": vuln_name,
                    "fix_name": fix_name,
                    "vuln_head": (r.get("vulnerable_func") or "")[:120],
                    "fix_head": (r.get("reference_fix") or "")[:120],
                })

    print(f"total records:                          {n}")
    print(f"  function name matches exactly:         {n_name_match} "
          f"({100*n_name_match/n:.1f}%)")
    print(f"  vuln func name appears somewhere in fix: {n_name_present_elsewhere} "
          f"({100*n_name_present_elsewhere/n:.1f}%)")
    print(f"  SUSPECT (name not found at all):       {n_suspect} "
          f"({100*n_suspect/n:.1f}%)")

    if suspects:
        print("\n--- first 5 suspect records ---")
        for s in suspects[:5]:
            print(f"\nid={s['id']}  vuln_name={s['vuln_name']!r}  fix_name={s['fix_name']!r}")
            print(f"  VULN: {s['vuln_head']!r}")
            print(f"  FIX : {s['fix_head']!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_reference_fix_quality.py diagnosis_records.jsonl")
        sys.exit(1)
    main(sys.argv[1])
