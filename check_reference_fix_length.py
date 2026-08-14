#!/usr/bin/env python3
"""
check_reference_fix_length.py

Flags reference_fix entries that are suspiciously long relative to
vulnerable_func -- a sign that reference_fix may contain extra
concatenated functions from the same commit, not just the one function
being repaired. This matters because generated_patch (from the LLM)
contains only the single target function, so comparing it against a
multi-function reference_fix would unfairly deflate exact-match /
CodeBLEU.

Usage:
    python check_reference_fix_length.py diagnosis_records.jsonl
"""

import json
import re
import sys


def normalize(code: str) -> str:
    if not code:
        return ""
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def main(path):
    ratios = []
    high = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = normalize(r.get("vulnerable_func", ""))
            fx = normalize(r.get("reference_fix", ""))
            if not v or not fx:
                continue
            ratio = len(fx) / len(v)
            ratios.append(ratio)
            if ratio > 2.0:
                high.append((r.get("id"), ratio, len(v), len(fx)))

    ratios.sort()
    n = len(ratios)
    if n == 0:
        print("no records with both fields populated")
        return

    def pct(p):
        return ratios[int(p * (n - 1))]

    print(f"n = {n}")
    print(f"median ratio (fix_len / vuln_len): {pct(0.5):.2f}")
    print(f"p90:  {pct(0.9):.2f}")
    print(f"p99:  {pct(0.99):.2f}")
    print(f"max:  {ratios[-1]:.2f}")
    print(f"\ncount with ratio > 2.0x: {len(high)} ({100*len(high)/n:.1f}%)")
    print(f"count with ratio > 3.0x: {sum(1 for h in high if h[1] > 3.0)}")

    if high:
        high.sort(key=lambda x: -x[1])
        print("\n--- top 5 highest-ratio records ---")
        for id_, ratio, vlen, flen in high[:5]:
            print(f"  id={id_}  ratio={ratio:.2f}  vuln_len={vlen}  fix_len={flen}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_reference_fix_length.py diagnosis_records.jsonl")
        sys.exit(1)
    main(sys.argv[1])
