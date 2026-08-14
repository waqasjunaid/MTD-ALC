#!/usr/bin/env python3
"""
compute_interrater_agreement.py

Computes Cohen's kappa and raw agreement between two raters' scores in
a completed expert_eval_sample.csv (from build_expert_eval_sample.py).

Usage:
    python compute_interrater_agreement.py expert_eval_sample.csv
"""
import csv
import sys
from collections import Counter


def cohens_kappa(rater1, rater2, categories):
    n = len(rater1)
    assert n == len(rater2)

    observed_agreement = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    p1 = Counter(rater1)
    p2 = Counter(rater2)
    expected_agreement = sum((p1[c] / n) * (p2[c] / n) for c in categories)

    if expected_agreement == 1.0:
        return 1.0, observed_agreement  # perfect agreement, no variance
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa, observed_agreement


def interpret_kappa(k):
    if k < 0: return "poor (worse than chance)"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    path = sys.argv[1]
    rater1, rater2 = [], []
    skipped = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r1 = row.get("rater1_score(1-4)", "").strip()
            r2 = row.get("rater2_score(1-4)", "").strip()
            if not r1 or not r2:
                skipped += 1
                continue
            rater1.append(int(r1))
            rater2.append(int(r2))

    if skipped:
        print(f"Skipped {skipped} rows with missing rater scores.")

    n = len(rater1)
    if n == 0:
        print("No fully-scored rows found -- both rater columns must be filled in.")
        return

    categories = sorted(set(rater1) | set(rater2))
    kappa, observed = cohens_kappa(rater1, rater2, categories)

    print(f"\nScored samples: {n}")
    print(f"Raw agreement: {observed:.3f} ({100*observed:.1f}%)")
    print(f"Cohen's kappa: {kappa:.3f} ({interpret_kappa(kappa)} agreement)")

    print(f"\nScore distribution:")
    print(f"  Rater 1: {dict(sorted(Counter(rater1).items()))}")
    print(f"  Rater 2: {dict(sorted(Counter(rater2).items()))}")

    print(f"\nFor the paper, report both raw agreement and kappa, e.g.:")
    print(f'  "Two independent raters scored {n} sampled patches using a '
          f'4-point rubric (Correct/Partial/Incorrect/Worsens), achieving '
          f'{100*observed:.1f}% raw agreement (Cohen\'s kappa = {kappa:.2f}, '
          f'{interpret_kappa(kappa)} agreement). Disagreements were resolved '
          f'by discussion."')


if __name__ == "__main__":
    main()
