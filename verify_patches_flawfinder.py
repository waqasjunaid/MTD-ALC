#!/usr/bin/env python3
"""
verify_patches_flawfinder.py

Independent verification of repair efficacy using Flawfinder, a standard
external C/C++ security static analyzer -- NOT your own trained MTD
model. This avoids two problems found with the MTD-based approach
(verify_patches.py):

  1. Circularity: verifying your model's repairs using the same model
     that flagged them isn't independently convincing to a reviewer.
  2. The 'heuristic' line-mapping strategy (required for any code with
     no ground-truth flaw-line labels, which includes every LLM-
     generated patch) was shown to have very weak discriminative power
     in your own detector -- vulnerable vs. benign BigVul samples
     scored under 'heuristic' strategy are nearly indistinguishable.

Flawfinder scores each function on rule-based hits (0-5 risk level per
hit, roughly aligned to CWE severity) independent of any of that.

Install:
    pip install flawfinder --break-system-packages

Usage:
    python verify_patches_flawfinder.py \
        --results results.jsonl \
        --out flawfinder_results.jsonl
"""

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Same annotation-stripping as verify_patches.py, so Flawfinder scores
# the actual code change, not the LLM's own "/* FIX [CWE-xxx] */" prose.
_INLINE_FIX_COMMENT_RE = re.compile(r"/\*\s*FIX\b.*?\*/", re.IGNORECASE | re.DOTALL)
_CHANGES_BLOCK_RE = re.compile(
    r"/\*\s*(CHANGES MADE|Changes made|Summary of changes)\s*:?.*?\*/",
    re.IGNORECASE | re.DOTALL,
)


def strip_llm_annotations(code: str) -> str:
    code = _CHANGES_BLOCK_RE.sub("", code)
    code = _INLINE_FIX_COMMENT_RE.sub("", code)
    return code


def check_flawfinder_available():
    if shutil.which("flawfinder") is None:
        print("ERROR: 'flawfinder' not found on PATH.")
        print("Install with: pip install flawfinder --break-system-packages")
        sys.exit(1)


def run_flawfinder(code: str, tmpdir: Path, name: str):
    """Writes code to a temp .c file, runs flawfinder --csv on it, returns
    (total_risk_score, hit_count, max_level, cwe_list)."""
    path = tmpdir / f"{name}.c"
    path.write_text(code, encoding="utf-8", errors="replace")

    proc = subprocess.run(
        ["flawfinder", "--csv", "--minlevel=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if not proc.stdout.strip():
        return 0, 0, 0, []

    reader = csv.DictReader(io.StringIO(proc.stdout))
    total_risk = 0
    hit_count = 0
    max_level = 0
    cwes = []
    for row in reader:
        try:
            level = int(row.get("Level", row.get("level", 0)))
        except (ValueError, TypeError):
            continue
        total_risk += level
        hit_count += 1
        max_level = max(max_level, level)
        cwe = row.get("CWEs", row.get("cwes", ""))
        if cwe:
            cwes.append(cwe)

    return total_risk, hit_count, max_level, cwes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    check_flawfinder_available()

    n = 0
    n_improved = 0        # risk score strictly decreased
    n_fully_clean = 0     # after has zero flawfinder hits
    n_worse = 0           # risk score increased
    n_unchanged = 0
    deltas = []

    with open(args.results) as fin, \
         tempfile.TemporaryDirectory() as tmpdir, \
         open(args.out, "w") as fout:

        tmpdir = Path(tmpdir)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec["id"])
            n += 1

            clean_patch = strip_llm_annotations(rec["generated_patch"])

            risk_before, hits_before, max_before, cwes_before = run_flawfinder(
                rec["vulnerable_func"], tmpdir, f"{sid}_before"
            )
            risk_after, hits_after, max_after, cwes_after = run_flawfinder(
                clean_patch, tmpdir, f"{sid}_after"
            )

            delta = risk_before - risk_after
            deltas.append(delta)

            if hits_after == 0:
                n_fully_clean += 1
            if delta > 0:
                n_improved += 1
            elif delta < 0:
                n_worse += 1
            else:
                n_unchanged += 1

            out_rec = {
                "id": sid,
                "risk_before": risk_before,
                "risk_after": risk_after,
                "delta": delta,
                "hits_before": hits_before,
                "hits_after": hits_after,
                "max_level_before": max_before,
                "max_level_after": max_after,
                "fully_clean_after": hits_after == 0,
            }
            fout.write(json.dumps(out_rec) + "\n")

    print(f"\ntotal patches:                {n}")
    print(f"risk score decreased:         {n_improved}/{n} ({100*n_improved/n:.1f}%)")
    print(f"risk score increased:         {n_worse}/{n} ({100*n_worse/n:.1f}%)")
    print(f"risk score unchanged:         {n_unchanged}/{n} ({100*n_unchanged/n:.1f}%)")
    print(f"fully clean after (0 hits):   {n_fully_clean}/{n} ({100*n_fully_clean/n:.1f}%)")
    print(f"mean risk delta (before-after): {sum(deltas)/len(deltas):.3f}")
    print(f"\nfull per-sample results written to {args.out}")


if __name__ == "__main__":
    main()
