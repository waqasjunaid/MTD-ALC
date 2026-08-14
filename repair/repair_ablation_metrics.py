#!/usr/bin/env python3
"""
repair_ablation_metrics.py

Computes correctness metrics (exact-match, CodeBLEU, construct-removal)
for repair patches, grouped by (dataset, arm).

Usage:
    python repair_ablation_metrics.py results.jsonl

Input format: one JSON object per line, with fields:
    sample_id        (str/int)
    dataset          (str, e.g. "bigvul")
    arm              (str, e.g. "diag")
    generated_patch  (str) - the repaired function
    reference_fix    (str) - the ground-truth fixed function
    vulnerable_func  (str) - the original vulnerable function

Install the real CodeBLEU implementation for the paper's numbers:
    pip install codebleu --break-system-packages
(falls back to an approximate BLEU-4 if unavailable, so the script
always runs, but the fallback should NOT be used for reported numbers).
"""

import json
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------
# Normalization: strip comments and collapse whitespace so that
# trivially different but identical code counts as an exact match.
# ---------------------------------------------------------------------
def normalize_code(code: str) -> str:
    if code is None:
        return ""
    # remove // line comments and /* */ block comments
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # collapse all runs of whitespace to a single space
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def exact_match(patch: str, reference: str) -> int:
    return int(normalize_code(patch) == normalize_code(reference)
               and normalize_code(patch) != "")


# ---------------------------------------------------------------------
# CodeBLEU. Uses the `codebleu` package if available; otherwise falls
# back to token-level BLEU-4 so the script still runs.
# ---------------------------------------------------------------------
def compute_codebleu(patch: str, reference: str) -> float:
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([reference], [patch], lang="c",
                                weights=(0.25, 0.25, 0.25, 0.25))
        return float(result["codebleu"])
    except Exception:
        return _fallback_bleu4(patch, reference)


def _fallback_bleu4(hyp: str, ref: str) -> float:
    import math
    hyp_t = normalize_code(hyp).split()
    ref_t = normalize_code(ref).split()
    if not hyp_t or not ref_t:
        return 0.0

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    precisions = []
    for n in range(1, 5):
        h = ngrams(hyp_t, n)
        r = ngrams(ref_t, n)
        if not h:
            precisions.append(0.0)
            continue
        r_counts = defaultdict(int)
        for g in r:
            r_counts[g] += 1
        overlap = 0
        h_counts = defaultdict(int)
        for g in h:
            h_counts[g] += 1
        for g, c in h_counts.items():
            overlap += min(c, r_counts.get(g, 0))
        precisions.append(overlap / len(h) if h else 0.0)

    if min(precisions) == 0:
        precisions = [max(p, 1e-9) for p in precisions]
    geo = math.exp(sum(math.log(p) for p in precisions) / 4)
    bp = 1.0 if len(hyp_t) > len(ref_t) else math.exp(1 - len(ref_t) / len(hyp_t))
    return bp * geo


# ---------------------------------------------------------------------
# Construct-removal check.
# This list is copied VERBATIM from mtd/task1_vulnerability_classification.py's
# _PATTERNS (minus the CWE tags, which aren't needed here) -- these are
# the exact regexes that drive your Task 1 score and therefore the V
# score. Using this list means "construct removed" means the same thing
# here as it does to your own detector. If task1's _PATTERNS ever
# changes, update this list to match.
# ---------------------------------------------------------------------
DANGEROUS_CONSTRUCTS = [
    r"\bstrcpy\s*\(",
    r"\bstrcat\s*\(",
    r"\bgets\s*\(",
    r"\bsprintf\s*\(",
    r"\bscanf\s*\(",
    r"\bmemcpy\s*\(",
    r"\bmemmove\s*\(",
    r"\balloca\s*\(",
    r"\bsnprintf\s*\(",
    r"\bprintf\s*\(\s*\w+\s*\)",
    r"\bfprintf\s*\(\s*\w+\s*,\s*\w+\s*\)",
    r"\bsyslog\s*\(\s*\w+\s*,\s*\w+\s*\)",
    r"\(int\)\s*strlen",
    r"<<\s*\d+",
    r"\batoi\s*\(",
    r"\batol\s*\(",
    r"\bmalloc\s*\(",
    r"\brealloc\s*\(",
    r"\bcalloc\s*\(",
    r"\bfree\s*\(\w+\)",
    r"\bsystem\s*\(",
    r"\bpopen\s*\(",
    r"\bexecve?\s*\(",
    r'"\.\./',
    r"\bopen\s*\(",
    r"\bfopen\s*\(",
    r"\brand\s*\(",
    r"\bsrand\s*\(\s*time",
    r"^\s*read\s*\(",
    r"^\s*write\s*\(",
    r"^\s*recv\s*\(",
    r"\w+\s*\[\s*\w+\s*\]",
    r"\(char\s*\*\)",
    r"\(void\s*\*\)",
    r"\+\+\s*\w+\s*\[",
]


def construct_removed(vulnerable: str, patch: str):
    """1 if a dangerous construct present in the vulnerable function is
    no longer present in the patch; 0 otherwise. If the vulnerable
    function contained no flagged construct, returns None (N/A)."""
    v = normalize_code(vulnerable)
    p = normalize_code(patch)
    present = [c for c in DANGEROUS_CONSTRUCTS if re.search(c, v)]
    if not present:
        return None  # not applicable to this sample
    still_there = [c for c in present if re.search(c, p)]
    return int(len(still_there) == 0)


def main(path):
    records = [json.loads(l) for l in open(path) if l.strip()]
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("dataset", "bigvul"), r.get("arm", "diag"))].append(r)

    print(f"{'dataset':<10} {'arm':<10} {'n':>5} "
          f"{'exact%':>8} {'codebleu':>9} {'constr_rm%':>11}")
    print("-" * 60)

    for (dataset, arm), recs in sorted(groups.items()):
        n = len(recs)
        em = sum(exact_match(r["generated_patch"], r["reference_fix"])
                  for r in recs)
        cb = [compute_codebleu(r["generated_patch"], r["reference_fix"])
              for r in recs]
        cr_vals = [construct_removed(r["vulnerable_func"], r["generated_patch"])
                   for r in recs]
        cr_applicable = [x for x in cr_vals if x is not None]

        em_pct = 100 * em / n if n else 0
        cb_mean = sum(cb) / len(cb) if cb else 0
        cr_pct = (100 * sum(cr_applicable) / len(cr_applicable)
                  if cr_applicable else 0)

        print(f"{dataset:<10} {arm:<10} {n:>5} "
              f"{em_pct:>7.1f}% {cb_mean:>9.3f} {cr_pct:>10.1f}%")
        print(f"  (construct-removal computed on {len(cr_applicable)}/{n} "
              f"samples that contained a flagged construct)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python repair_ablation_metrics.py results.jsonl")
        sys.exit(1)
    main(sys.argv[1])