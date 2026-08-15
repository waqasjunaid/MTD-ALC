#!/usr/bin/env python3
"""
mcnemar_test.py

Paired significance test for the diag-vs-cwe_only ablation. Since both
arms were scored on the SAME 548 samples, this is paired binary data
(construct removed: yes/no, per sample, per arm) -- the correct test is
McNemar's exact test, not an independent two-sample comparison.

Only samples where the vulnerable function actually contains a flagged
construct are included (construct-removal is undefined otherwise) --
this reproduces the same 142-sample denominator repair_ablation_metrics.py
already reported, using the identical DANGEROUS_CONSTRUCTS list.

Usage:
    python mcnemar_test.py \
        --diag results.jsonl \
        --cwe_only results_armB.jsonl
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

# Import the exact same construct list and normalize_code() used to
# produce the 142-sample denominator, so this test is consistent with
# the reported percentages.
sys.path.insert(0, str(Path(__file__).resolve().parent / "repair"))
try:
    from repair_ablation_metrics import DANGEROUS_CONSTRUCTS, normalize_code
except ImportError:
    print("ERROR: could not import repair_ablation_metrics.py -- make sure "
          "this script sits next to (or 'repair/' contains) that file.")
    sys.exit(1)

_PATTERNS = [re.compile(p) for p in DANGEROUS_CONSTRUCTS]


def has_construct(code: str) -> bool:
    v = normalize_code(code)
    return any(p.search(v) for p in _PATTERNS)


def construct_removed(vulnerable: str, patch: str) -> bool:
    v = normalize_code(vulnerable)
    p = normalize_code(patch)
    present = [pat for pat in _PATTERNS if pat.search(v)]
    if not present:
        return None
    return not any(pat.search(p) for pat in present)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar test (sign-test formulation): under H0,
    each discordant pair is a coin flip, so b ~ Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: 2 * P(X <= k) for X ~ Binomial(n, 0.5), capped at 1.0
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--cwe_only", required=True)
    args = ap.parse_args()

    diag = {json.loads(l)["id"]: json.loads(l) for l in open(args.diag) if l.strip()}
    cwe = {json.loads(l)["id"]: json.loads(l) for l in open(args.cwe_only) if l.strip()}

    common_ids = sorted(set(diag) & set(cwe))
    print(f"samples present in both arms: {len(common_ids)}")

    both_removed = only_diag = only_cwe = neither = 0
    applicable = 0

    for sid in common_ids:
        vuln = diag[sid]["vulnerable_func"]  # same original function in both arms
        if not has_construct(vuln):
            continue
        applicable += 1

        diag_removed = construct_removed(vuln, diag[sid]["generated_patch"])
        cwe_removed = construct_removed(vuln, cwe[sid]["generated_patch"])

        if diag_removed and cwe_removed:
            both_removed += 1
        elif diag_removed and not cwe_removed:
            only_diag += 1
        elif cwe_removed and not diag_removed:
            only_cwe += 1
        else:
            neither += 1

    print(f"applicable (construct present in original): {applicable}")
    print(f"\n2x2 contingency table (construct removed?):")
    print(f"                    cwe_only: yes   cwe_only: no")
    print(f"  diag: yes         {both_removed:>13}   {only_diag:>13}")
    print(f"  diag: no          {only_cwe:>13}   {neither:>13}")

    b, c = only_diag, only_cwe
    p_value = exact_mcnemar_pvalue(b, c)

    print(f"\nDiscordant pairs: diag-only={b}, cwe_only-only={c}  "
          f"(total discordant n={b+c})")
    print(f"McNemar's exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("-> Statistically significant at alpha=0.05.")
    else:
        print("-> NOT statistically significant at alpha=0.05. "
              "The observed difference is consistent with chance.")


if __name__ == "__main__":
    main()
