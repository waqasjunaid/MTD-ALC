#!/usr/bin/env python3
"""
diagnose_verify_anomaly.py

verify_patches.py reported 513/548 (93.6%) "fixed" and 0/548 still
flagged vulnerable -- but repair_ablation_metrics.py separately found
only 45.5% of construct-flagged samples had the dangerous construct
actually removed. Those two findings are inconsistent if V_after is
behaving sensibly. This script finds the overlap and shows exactly
what's happening on the samples where the contradiction should be
sharpest: patches that STILL contain a flagged dangerous construct,
but were nonetheless marked "self-verified fixed".

Run from the project root (same place you ran verify_patches.py).

Usage:
    python diagnose_verify_anomaly.py \
        --results results.jsonl \
        --verify verify_results.jsonl \
        --srcdir verify_sources
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mtd"))
sys.path.insert(0, str(ROOT / "mtd" / "ml"))

import task1_vulnerability_classification as task1

# same list task1.py actually uses -- copied for direct inspection here
_PATTERNS = [p for p, _ in task1._PATTERNS] if hasattr(task1, "_PATTERNS") else []


# Severe subset only -- excludes weak/generic patterns (array indexing,
# casts, bit shift) that are common in ALL real C code and carry low
# confidence (0.40-0.55) in suspicious_line_mapper.py's own rules. This
# is the strict test: does a genuinely dangerous, high-confidence call
# (confidence >= 0.65 in your own rules) still get scored as "fixed"?
_SEVERE_PATTERNS = [
    re.compile(p) for p in [
        r"\bstrcpy\s*\(", r"\bstrcat\s*\(", r"\bgets\s*\(",
        r"\bsprintf\s*\(", r"\bscanf\s*\(", r"\bmemcpy\s*\(",
        r"\bmemmove\s*\(", r"\balloca\s*\(",
        r"\bmalloc\s*\(", r"\brealloc\s*\(", r"\bcalloc\s*\(",
        r"\bsystem\s*\(", r"\bpopen\s*\(", r"\bexecve?\s*\(",
    ]
]


def has_severe_construct(code: str) -> bool:
    return any(p.search(code) for p in _SEVERE_PATTERNS)


def has_construct(code: str) -> bool:
    return any(p.search(code) for p in _PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--srcdir", default="verify_sources")
    args = ap.parse_args()

    results = {json.loads(l)["id"]: json.loads(l) for l in open(args.results) if l.strip()}
    verify = {json.loads(l)["id"]: json.loads(l) for l in open(args.verify) if l.strip()}
    srcdir = Path(args.srcdir)

    contradictions = []
    severe_contradictions = []
    for sid, v in verify.items():
        r = results.get(sid)
        if r is None:
            continue
        patch = r["generated_patch"]
        is_fixed = v["was_vulnerable_fair"] and not v["still_vulnerable_fair"]
        if has_construct(patch) and is_fixed:
            contradictions.append((sid, v))
        if has_severe_construct(patch) and is_fixed:
            severe_contradictions.append((sid, v))

    print(f"Total scored: {len(verify)}")
    print(f"Patches with ANY Task1-flagged construct (broad list): "
          f"{sum(1 for r in results.values() if has_construct(r['generated_patch']))}")
    print(f"  -> marked 'fixed' anyway: {len(contradictions)}")
    print(f"Patches with a SEVERE construct still present "
          f"(strcpy/gets/system/memcpy/etc.): "
          f"{sum(1 for r in results.values() if has_severe_construct(r['generated_patch']))}")
    print(f"  -> marked 'fixed' anyway: {len(severe_contradictions)}  <-- the real test")

    if not severe_contradictions:
        print("\nNo severe-construct contradictions -- the verification result "
              "holds up even under the strict test. Good sign.")
        return

    print("\n--- inspecting first 3 SEVERE contradiction cases directly ---")
    for sid, v in severe_contradictions[:3]:
        src_path = srcdir / f"{sid}.c"
        code = src_path.read_text(encoding="utf-8", errors="replace") if src_path.exists() else "<file missing>"

        print(f"\n{'='*70}\nid={sid}  V_before_heuristic={v['V_before_heuristic']}  "
              f"V_after={v['V_after']}  delta={v['delta_fair']}")

        hits = [p.pattern for p in _PATTERNS if p.search(code)]
        print(f"Task1 patterns that match the PERSISTED verify_sources/{sid}.c: {hits}")

        # Directly call task1.run on this exact file/func to see its raw score
        func = {"code": code, "line_count": len(code.splitlines()), "param_count": 0}
        r1 = task1.run(str(src_path), [], func, {"strategy": "heuristic", "suspicious_lines": []})
        print(f"task1.run() with EMPTY suspicious_lines: score={r1['score']}  "
              f"features={r1['features']}")


if __name__ == "__main__":
    main()