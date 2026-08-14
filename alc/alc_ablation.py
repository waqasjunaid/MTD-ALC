#!/usr/bin/env python3
"""
alc_ablation.py

Computes the ALC component ablation: what would be reported to a user
WITHOUT ALC (every MTD-VULNERABLE prediction reported directly) versus
WITH ALC (only TRUSTWORTHY predictions reported directly; UNTRUSTWORTHY
predictions -- including missed vulnerabilities ALC flags for review --
are deferred to Diagnosis rather than silently accepted or reported
unfiltered).

This is computed directly from the live bigvul_results.csv and
megavul_results.csv files (not recalled from manuscript prose), so the
numbers are guaranteed current.

Usage:
    python alc_ablation.py --bigvul bigvul_results.csv --megavul megavul_results.csv
"""
import argparse
import csv


def compute(path, dataset_name):
    tp_trust = fp_trust = fn_trust = tn_trust = 0
    tp_untrust = fp_untrust = fn_untrust = tn_untrust = 0

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = int(row["label"])
            verdict = row["mtd_verdict"]  # VULNERABLE / NON_VULNERABLE
            decision = row["alc_decision"]  # trustworthy / untrustworthy

            is_trust = (decision.lower() == "trustworthy")
            is_vuln_pred = (verdict.upper() == "VULNERABLE")

            if is_vuln_pred and label == 1:
                if is_trust: tp_trust += 1
                else: tp_untrust += 1
            elif is_vuln_pred and label == 0:
                if is_trust: fp_trust += 1
                else: fp_untrust += 1
            elif not is_vuln_pred and label == 1:
                if is_trust: fn_trust += 1
                else: fn_untrust += 1
            else:
                if is_trust: tn_trust += 1
                else: tn_untrust += 1

    total_vuln_pred = tp_trust + tp_untrust + fp_trust + fp_untrust
    total_fp = fp_trust + fp_untrust
    total_tn = tn_trust + tn_untrust
    total_fn = fn_trust + fn_untrust

    # WITHOUT ALC: every MTD-VULNERABLE prediction is reported directly,
    # regardless of trust decision. All FN are silently accepted (never
    # flagged for review at all, since ALC doesn't exist in this scenario).
    without_precision = (tp_trust + tp_untrust) / total_vuln_pred if total_vuln_pred else 0.0
    without_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) else 0.0
    without_fn_silently_missed = total_fn  # ALL of them, no ALC to flag any

    # WITH ALC: only TRUSTWORTHY predictions reported directly.
    reported_positives = tp_trust + fp_trust
    with_precision = tp_trust / reported_positives if reported_positives else 0.0
    with_fpr = fp_trust / (fp_trust + tn_trust) if (fp_trust + tn_trust) else 0.0
    with_fn_silently_missed = fn_trust  # fn_untrust get flagged for review instead

    print(f"\n===== {dataset_name} =====")
    print(f"Total MTD-VULNERABLE predictions: {total_vuln_pred}")
    print(f"Total false positives (FP): {total_fp}  (trust={fp_trust}, untrust={fp_untrust})")
    print(f"Total false negatives (FN): {total_fn}  (trust={fn_trust}, untrust={fn_untrust})")

    print(f"\n--- WITHOUT ALC (report every MTD-VULNERABLE prediction directly) ---")
    print(f"  Reported precision: {without_precision:.4f}")
    print(f"  Reported FPR:       {without_fpr:.6f}")
    print(f"  Missed vulnerabilities silently passed (no review at all): "
          f"{without_fn_silently_missed}/{total_fn} (100%)")

    print(f"\n--- WITH ALC (only TRUSTWORTHY reported directly) ---")
    print(f"  Reported precision: {with_precision:.4f}")
    print(f"  Reported FPR:       {with_fpr:.6f}")
    pct_fn_still_missed = 100 * with_fn_silently_missed / total_fn if total_fn else 0
    print(f"  Missed vulnerabilities silently passed: "
          f"{with_fn_silently_missed}/{total_fn} ({pct_fn_still_missed:.1f}%)")
    print(f"  Missed vulnerabilities FLAGGED for review by ALC: "
          f"{fn_untrust}/{total_fn} ({100 - pct_fn_still_missed:.1f}%)")

    print(f"\n--- Improvement from ALC ---")
    prec_mult = (with_precision / without_precision) if without_precision else float("nan")
    fpr_div = (without_fpr / with_fpr) if with_fpr else float("nan")
    print(f"  Precision multiplier: {prec_mult:.2f}x")
    print(f"  FPR reduction factor: {fpr_div:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigvul", required=True)
    ap.add_argument("--megavul", required=True)
    args = ap.parse_args()

    compute(args.bigvul, "BigVul")
    compute(args.megavul, "MegaVul")


if __name__ == "__main__":
    main()
