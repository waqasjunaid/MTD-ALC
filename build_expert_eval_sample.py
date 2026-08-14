#!/usr/bin/env python3
"""
build_expert_eval_sample.py

Builds a representative sample of repair patches for expert human
evaluation -- the specific remedy the reviewer suggested ("expert
evaluation of a representative subset of examples").

Stratifies the sample across:
  - whether the vulnerable function contains a flagged dangerous
    construct (same regex list as repair_ablation_metrics.py), so the
    sample isn't dominated by either "obviously security-relevant" or
    "less obviously flagged" cases
  - construct-removal outcome (removed / not removed), where applicable

Outputs a CSV with one row per sampled patch, ready to open in a
spreadsheet: vulnerable function, generated patch (header/annotation
comments stripped for readability), reference fix (for context only --
raters should judge on security merit, not just similarity to this),
and empty columns for two independent raters to fill in.

Rubric (documented in the output file's first row as a comment, and
printed to console):
    1 = Correct: identifies and fixes the actual vulnerability
    2 = Partial: addresses a related but incomplete/cosmetic aspect
    3 = Incorrect/Misattributed: does not address the actual
        vulnerability (may hardens an unrelated pattern instead)
    4 = Worsens: introduces a new risk or breaks functionality

Usage:
    python build_expert_eval_sample.py --results results.jsonl \
        --n 30 --out expert_eval_sample.csv
"""
import argparse
import csv
import json
import random
import re

_INLINE_FIX_COMMENT_RE = re.compile(r"/\*\s*FIX\b.*?\*/", re.IGNORECASE | re.DOTALL)
_HEADER_RE = re.compile(r"^\s*/\*.*?Repaired by LLM Repair Module.*?\*/\s*", re.DOTALL)


def strip_leading_block_comment(text):
    stripped = text.lstrip()
    if not stripped.startswith("/*"):
        return text
    end = stripped.find("*/")
    if end == -1:
        return text
    return stripped[end + 2:].lstrip()


def clean_patch(text):
    text = _HEADER_RE.sub("", text)
    text = strip_leading_block_comment(text)
    text = _INLINE_FIX_COMMENT_RE.sub("", text)
    return text.strip()


DANGEROUS_CONSTRUCTS = [
    r"\bstrcpy\s*\(", r"\bstrcat\s*\(", r"\bgets\s*\(", r"\bsprintf\s*\(",
    r"\bscanf\s*\(", r"\bmemcpy\s*\(", r"\bmemmove\s*\(", r"\balloca\s*\(",
    r"\bmalloc\s*\(", r"\brealloc\s*\(", r"\bcalloc\s*\(", r"\bsystem\s*\(",
    r"\bpopen\s*\(", r"\bexecve?\s*\(",
]
_PATTERNS = [re.compile(p) for p in DANGEROUS_CONSTRUCTS]


def has_construct(code):
    return any(p.search(code) for p in _PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.results) if l.strip()]

    with_construct = [r for r in recs if has_construct(r["vulnerable_func"])]
    without_construct = [r for r in recs if not has_construct(r["vulnerable_func"])]

    rng = random.Random(args.seed)
    n_with = min(len(with_construct), args.n // 2)
    n_without = min(len(without_construct), args.n - n_with)

    sample = rng.sample(with_construct, n_with) + rng.sample(without_construct, n_without)
    rng.shuffle(sample)

    print(f"Rubric:")
    print(f"  1 = Correct: identifies and fixes the actual vulnerability")
    print(f"  2 = Partial: addresses a related but incomplete/cosmetic aspect")
    print(f"  3 = Incorrect/Misattributed: does not address the actual vulnerability")
    print(f"  4 = Worsens: introduces a new risk or breaks functionality")
    print()
    print(f"Sampled {len(sample)} patches ({n_with} construct-flagged, "
          f"{n_without} not flagged) into {args.out}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "had_flagged_construct", "vulnerable_func",
            "generated_patch", "reference_fix_for_context",
            "rater1_score(1-4)", "rater1_notes",
            "rater2_score(1-4)", "rater2_notes",
        ])
        for r in sample:
            writer.writerow([
                r["id"],
                has_construct(r["vulnerable_func"]),
                r["vulnerable_func"],
                clean_patch(r["generated_patch"]),
                r["reference_fix"],
                "", "", "", "",
            ])

    print(f"\nNext steps:")
    print(f"  1. Open {args.out} in a spreadsheet application.")
    print(f"  2. Have two independent raters (ideally without seeing each")
    print(f"     other's scores) fill in rater1_score/rater2_score per the rubric.")
    print(f"  3. Once both are filled in, run compute_interrater_agreement.py")
    print(f"     on the completed file.")


if __name__ == "__main__":
    main()
