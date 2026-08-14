#!/usr/bin/env python3
"""
analyze_flawfinder_results.py

verify_patches_flawfinder.py's headline numbers can be misleading if
Flawfinder found nothing in most ORIGINAL vulnerable functions either
(0 hits -> 0 hits counts as both "unchanged" and "fully clean after").
This recomputes the meaningful, conditional numbers: performance only
on samples where Flawfinder actually detected something in the
original code.

Usage:
    python analyze_flawfinder_results.py flawfinder_results.jsonl
"""
import json
import sys


def main(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    n = len(recs)

    had_risk_before = [r for r in recs if r["hits_before"] > 0]
    no_risk_before = [r for r in recs if r["hits_before"] == 0]

    print(f"total samples: {n}")
    print(f"Flawfinder found NOTHING in the original vulnerable function: "
          f"{len(no_risk_before)} ({100*len(no_risk_before)/n:.1f}%)")
    print(f"Flawfinder found >=1 issue in the original vulnerable function: "
          f"{len(had_risk_before)} ({100*len(had_risk_before)/n:.1f}%)  <-- the real denominator")

    if not had_risk_before:
        return

    m = len(had_risk_before)
    improved = sum(1 for r in had_risk_before if r["delta"] > 0)
    worse = sum(1 for r in had_risk_before if r["delta"] < 0)
    unchanged = sum(1 for r in had_risk_before if r["delta"] == 0)
    clean_after = sum(1 for r in had_risk_before if r["hits_after"] == 0)
    mean_delta = sum(r["delta"] for r in had_risk_before) / m

    print(f"\n--- of the {m} samples Flawfinder actually flagged originally ---")
    print(f"risk score decreased:   {improved}/{m} ({100*improved/m:.1f}%)")
    print(f"risk score increased:   {worse}/{m} ({100*worse/m:.1f}%)")
    print(f"risk score unchanged:   {unchanged}/{m} ({100*unchanged/m:.1f}%)")
    print(f"fully clean after:      {clean_after}/{m} ({100*clean_after/m:.1f}%)")
    print(f"mean risk delta:        {mean_delta:.3f}")

    print(f"\n--- first 5 'unchanged' cases among the flagged {m} (worth eyeballing) ---")
    shown = 0
    for r in had_risk_before:
        if r["delta"] == 0:
            print(f"  id={r['id']}  hits_before={r['hits_before']}  "
                  f"hits_after={r['hits_after']}  risk_before={r['risk_before']} "
                  f"risk_after={r['risk_after']}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python analyze_flawfinder_results.py flawfinder_results.jsonl")
        sys.exit(1)
    main(sys.argv[1])
