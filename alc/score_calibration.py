#!/usr/bin/env python3
"""
score_calibration.py

Compute standard calibration metrics AND selective-prediction metrics from an
existing results CSV (e.g. bigvul_results.csv). No pipeline re-run needed.

Answers the reviewer request for ECE / Brier / reliability diagrams, and adds the
selective-prediction view (risk-coverage curve + the ALC operating point), which is
what the ALC trust score actually governs.

Standard calibration (on the MTD probability V_score vs. the true label):
  * Brier score          -- mean squared error of V against the 0/1 label
  * ECE / MCE            -- expected / max calibration error over probability bins
  * reliability diagram  -- binned mean(V) vs. empirical positive rate (+PNG)

Selective prediction (on the ALC trust score T_score):
  * risk-coverage curve + AURC -- does error on the retained set fall as we defer
    low-trust predictions? (+PNG)
  * ALC operating point        -- accuracy of TRUSTWORTHY vs. UNTRUSTWORTHY sets,
    computed directly from the alc_decision column

Usage:
  python score_calibration.py --csv bigvul_results.csv --out calibration.json
Columns expected: label, V_score, T_score, mtd_verdict, alc_decision
"""

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load(path, cols):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        missing = [c for c in cols if c not in rd.fieldnames]
        if missing:
            sys.exit(f"ERROR: CSV missing columns {missing}. Found: {rd.fieldnames}")
        for r in rd:
            try:
                rows.append({
                    "label": int(float(r["label"])),
                    "V": float(r["V_score"]),
                    "T": float(r["T_score"]) if r.get("T_score", "") not in ("", None) else None,
                    "verdict": 1 if str(r["mtd_verdict"]).strip().upper() == "VULNERABLE" else 0,
                    "alc": str(r.get("alc_decision", "")).strip().lower(),
                })
            except (ValueError, KeyError):
                continue
    return rows


def brier(rows):
    return sum((r["V"] - r["label"]) ** 2 for r in rows) / len(rows)


def reliability(rows, n_bins=10):
    """Bin by V; compare mean(V) to empirical positive rate. Returns bins + ECE/MCE."""
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        b = min(n_bins - 1, int(r["V"] * n_bins))
        bins[b].append(r)
    N = len(rows)
    table, ece, mce = [], 0.0, 0.0
    for i, b in enumerate(bins):
        if not b:
            table.append({"bin": i, "lo": i / n_bins, "hi": (i + 1) / n_bins,
                          "n": 0, "mean_V": None, "pos_rate": None, "gap": None})
            continue
        mean_V = sum(x["V"] for x in b) / len(b)
        pos = sum(x["label"] for x in b) / len(b)
        gap = abs(mean_V - pos)
        ece += (len(b) / N) * gap
        mce = max(mce, gap)
        table.append({"bin": i, "lo": i / n_bins, "hi": (i + 1) / n_bins,
                      "n": len(b), "mean_V": round(mean_V, 4),
                      "pos_rate": round(pos, 4), "gap": round(gap, 4)})
    return table, ece, mce


def risk_coverage(rows, steps=50):
    """Sort by trust T descending; report error rate on the top-coverage fraction.
    AURC = area under the risk-coverage curve (lower is better)."""
    have_T = [r for r in rows if r["T"] is not None]
    if not have_T:
        return None, None, "no T_score column"
    ordered = sorted(have_T, key=lambda r: r["T"], reverse=True)
    N = len(ordered)
    curve, cum_err = [], 0
    # precompute cumulative errors for exact risk at each prefix
    prefix_err = []
    e = 0
    for r in ordered:
        e += 0 if r["verdict"] == r["label"] else 1
        prefix_err.append(e)
    for s in range(1, steps + 1):
        k = max(1, round(N * s / steps))
        risk = prefix_err[k - 1] / k
        curve.append({"coverage": round(k / N, 4), "risk": round(risk, 4), "kept": k})
    # AURC: mean risk over all prefixes (standard discretization)
    aurc = sum(prefix_err[i] / (i + 1) for i in range(N)) / N
    return curve, aurc, None


def alc_operating_point(rows):
    trust = [r for r in rows if r["alc"] == "trustworthy"]
    untr = [r for r in rows if r["alc"] == "untrustworthy"]
    def acc(g): return sum(1 for r in g if r["verdict"] == r["label"]) / len(g) if g else None
    def err(g): return (1 - acc(g)) if g else None
    N = len(rows)
    return {
        "trustworthy_n": len(trust), "untrustworthy_n": len(untr),
        "coverage_trustworthy": round(len(trust) / N, 4) if N else 0,
        "acc_trustworthy": round(acc(trust), 4) if trust else None,
        "acc_untrustworthy": round(acc(untr), 4) if untr else None,
        "err_trustworthy": round(err(trust), 4) if trust else None,
        "err_untrustworthy": round(err(untr), 4) if untr else None,
    }


def maybe_plot_reliability(table, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [t["mean_V"] for t in table if t["mean_V"] is not None]
        ys = [t["pos_rate"] for t in table if t["pos_rate"] is not None]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#1f4e79", label="MTD (V_score)")
        ax.set_xlabel("mean predicted P(vulnerable)")
        ax.set_ylabel("empirical positive rate")
        ax.set_title("Reliability diagram — BigVul")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return path
    except Exception as e:
        return f"(matplotlib unavailable: {e})"


def maybe_plot_rc(curve, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [c["coverage"] for c in curve]; ys = [c["risk"] for c in curve]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(xs, ys, "-", color="#a83232")
        ax.set_xlabel("coverage (fraction retained, highest-trust first)")
        ax.set_ylabel("selective risk (error rate on retained)")
        ax.set_title("Risk-coverage — ALC trust score")
        ax.set_xlim(0, 1); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        return path
    except Exception as e:
        return f"(matplotlib unavailable: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--fig_prefix", default="calib")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"ERROR: not found: {args.csv}")
    rows = load(args.csv, ["label", "V_score", "mtd_verdict"])
    if not rows:
        sys.exit("ERROR: no usable rows.")
    print(f"loaded {len(rows)} rows")

    b = brier(rows)
    table, ece, mce = reliability(rows, args.n_bins)
    curve, aurc, rc_note = risk_coverage(rows)
    op = alc_operating_point(rows)

    print("\n=== STANDARD CALIBRATION (V_score vs. label) ===")
    print(f"  Brier score : {b:.4f}   (0 = perfect; lower is better)")
    print(f"  ECE         : {ece:.4f}  ({args.n_bins} bins)")
    print(f"  MCE         : {mce:.4f}")
    print("  reliability diagram (per bin):")
    print("    bin   range        n      mean_V   pos_rate  gap")
    for t in table:
        if t["n"] == 0:
            continue
        print(f"    {t['bin']:>2}  [{t['lo']:.1f},{t['hi']:.1f})  {t['n']:>6}   "
              f"{t['mean_V']:.4f}   {t['pos_rate']:.4f}   {t['gap']:.4f}")

    print("\n=== SELECTIVE PREDICTION (ALC trust score) ===")
    if curve:
        print(f"  AURC (area under risk-coverage): {aurc:.4f}  (lower = trust ranks errors well)")
        # a few readable points
        for target in (0.5, 0.7, 0.9, 1.0):
            pt = min(curve, key=lambda c: abs(c["coverage"] - target))
            print(f"    coverage {pt['coverage']:.2f}: selective risk {pt['risk']:.4f}")
    else:
        print(f"  risk-coverage: {rc_note}")

    print("\n=== ALC OPERATING POINT (from alc_decision) ===")
    print(f"  TRUSTWORTHY  : n={op['trustworthy_n']} (coverage {op['coverage_trustworthy']:.3f}) "
          f"accuracy {op['acc_trustworthy']}")
    print(f"  UNTRUSTWORTHY: n={op['untrustworthy_n']} accuracy {op['acc_untrustworthy']}")
    if op["acc_trustworthy"] is not None and op["acc_untrustworthy"] is not None:
        print(f"  -> retained (TRUSTWORTHY) error {op['err_trustworthy']} vs "
              f"deferred (UNTRUSTWORTHY) error {op['err_untrustworthy']}: "
              f"{'ALC defers the harder cases' if op['err_untrustworthy'] > op['err_trustworthy'] else 'check'}")

    rel_png = maybe_plot_reliability(table, f"{args.fig_prefix}_reliability.png")
    rc_png = maybe_plot_rc(curve, f"{args.fig_prefix}_risk_coverage.png") if curve else "(no curve)"
    print(f"\nfigures: {rel_png} | {rc_png}")

    Path(args.out).write_text(json.dumps({
        "n": len(rows), "brier": b, "ece": ece, "mce": mce, "n_bins": args.n_bins,
        "reliability_bins": table, "aurc": aurc, "risk_coverage": curve,
        "alc_operating_point": op,
    }, indent=2))
    print(f"saved {args.out}")
    print("\nNote: Brier/ECE/reliability assess the MTD probability V_score. The")
    print("risk-coverage curve and ALC operating point assess the ALC trust score as")
    print("a selective predictor, which is what ALC actually governs (routing, not")
    print("probability recalibration). Report both; frame ALC as selective prediction.")


if __name__ == "__main__":
    main()
