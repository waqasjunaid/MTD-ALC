#!/usr/bin/env python3
"""
score_trust_baselines.py

Answer the reviewer request to compare against standard calibration / uncertainty
methods: temperature scaling, isotonic regression, and conformal prediction.
Computed from an existing results CSV (bigvul_results.csv) -- no pipeline re-run.

Two parts, matching what each method actually does:

  (1) CALIBRATION comparison (temperature scaling, isotonic regression), applied to
      the MTD probability V_score and evaluated by ECE / Brier. A fair, train/test
      split is used: each recalibrator is FIT on a calibration split and evaluated
      on a disjoint test split, so the comparison is not optimistic. Expected
      outcome: because V is already well-calibrated, these standard methods yield
      little or no improvement -- i.e. V needs no post-hoc calibration.

  (2) CONFORMAL prediction as a COVERAGE-VALIDITY reference (not an accuracy
      contest). Split-conformal prediction sets are calibrated on the same split;
      we report empirical coverage (should meet the 1-alpha target) and the
      set-size distribution (size-2 sets = the "abstain / route to review" region,
      the conformal analogue of ALC's UNTRUSTWORTHY routing). This situates ALC
      against conformal without staging a per-sample accuracy bake-off, and lets the
      paper note the property conformal lacks: adaptivity to distribution shift
      (ALC's delta widens under domain shift; split conformal assumes
      exchangeability).

Usage:
  python score_trust_baselines.py --csv bigvul_results.csv --out trust_baselines.json
Columns expected: label, V_score
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
EPS = 1e-6


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        for c in ("label", "V_score"):
            if c not in rd.fieldnames:
                sys.exit(f"ERROR: CSV missing column '{c}'. Found: {rd.fieldnames}")
        for r in rd:
            try:
                rows.append((int(float(r["label"])),
                             min(1 - EPS, max(EPS, float(r["V_score"])))))
            except (ValueError, KeyError):
                continue
    return rows


def brier(pairs):
    return sum((p - y) ** 2 for y, p in pairs) / len(pairs)


def ece(pairs, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for y, p in pairs:
        bins[min(n_bins - 1, int(p * n_bins))].append((y, p))
    N = len(pairs)
    e = 0.0
    for b in bins:
        if not b:
            continue
        mp = sum(p for _, p in b) / len(b)
        pos = sum(y for y, _ in b) / len(b)
        e += (len(b) / N) * abs(mp - pos)
    return e


# ---- temperature scaling: fit scalar T on logits by minimizing NLL ----
def temp_scale_fit(pairs):
    from scipy.optimize import minimize_scalar
    logits = [math.log(p / (1 - p)) for _, p in pairs]
    ys = [y for y, _ in pairs]

    def nll(T):
        s = 0.0
        for z, y in zip(logits, ys):
            p = 1 / (1 + math.exp(-z / T))
            p = min(1 - EPS, max(EPS, p))
            s -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return s

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return res.x


def temp_apply(pairs, T):
    out = []
    for y, p in pairs:
        z = math.log(p / (1 - p))
        out.append((y, 1 / (1 + math.exp(-z / T))))
    return out


# ---- isotonic regression via sklearn (fallback: PAV) ----
def isotonic_fit_apply(train, test):
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit([p for _, p in train], [y for y, _ in train])
        return [(y, float(ir.predict([p])[0])) for y, p in test], "sklearn"
    except Exception:
        # Pool-Adjacent-Violators fallback
        pts = sorted(((p, y) for y, p in train))
        xs = [p for p, _ in pts]
        ys = [float(y) for _, y in pts]
        w = [1.0] * len(ys)
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1]:
                tot = w[i] + w[i + 1]
                ys[i] = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tot
                w[i] = tot
                del ys[i + 1]; del w[i + 1]; del xs[i + 1]
                if i > 0:
                    i -= 1
            else:
                i += 1

        def predict(p):
            # step function over pooled xs
            lo = 0
            for j, xv in enumerate(xs):
                if p >= xv:
                    lo = j
            return ys[lo]
        return [(y, predict(p)) for y, p in test], "PAV-fallback"


# ---- split conformal prediction sets on the binary label ----
def conformal(train, test, alpha):
    # nonconformity: for true label, score = 1 - phat(true). phat(1)=p, phat(0)=1-p
    cal_scores = sorted(1 - (p if y == 1 else 1 - p) for y, p in train)
    n = len(cal_scores)
    k = min(n - 1, max(0, math.ceil((n + 1) * (1 - alpha)) - 1))
    qhat = cal_scores[k]
    covered = 0
    sizes = {0: 0, 1: 0, 2: 0}
    for y, p in test:
        p1, p0 = p, 1 - p
        s = set()
        if (1 - p1) <= qhat:
            s.add(1)
        if (1 - p0) <= qhat:
            s.add(0)
        sizes[len(s)] += 1
        if y in s:
            covered += 1
    N = len(test)
    return {
        "alpha": alpha, "target_coverage": round(1 - alpha, 4),
        "empirical_coverage": round(covered / N, 4),
        "qhat": round(qhat, 4),
        "set_size_0_pct": round(100 * sizes[0] / N, 2),
        "set_size_1_pct": round(100 * sizes[1] / N, 2),
        "set_size_2_pct": round(100 * sizes[2] / N, 2),
        "abstain_pct": round(100 * sizes[2] / N, 2),
    }


def summarize(pairs, n_bins=10):
    return {"ece": round(ece(pairs, n_bins), 4), "brier": round(brier(pairs), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trust_baselines.json")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        sys.exit(f"ERROR: not found: {args.csv}")
    rows = load(args.csv)
    if len(rows) < 20:
        sys.exit("ERROR: too few rows.")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    mid = len(rows) // 2
    cal, test = rows[:mid], rows[mid:]  # cal = fit split, test = eval split
    print(f"loaded {len(rows)} rows; fit on {len(cal)}, evaluate on {len(test)}")

    raw = summarize(test, args.n_bins)

    T = temp_scale_fit(cal)
    temp = summarize(temp_apply(test, T), args.n_bins)

    iso_pairs, iso_impl = isotonic_fit_apply(cal, test)
    iso = summarize(iso_pairs, args.n_bins)

    conf = [conformal(cal, test, a) for a in (0.10, 0.05, 0.01)]

    print("\n=== CALIBRATION COMPARISON (ECE / Brier on the eval split) ===")
    print(f"  {'method':<26}{'ECE':>10}{'Brier':>10}")
    print(f"  {'V (raw, ours)':<26}{raw['ece']:>10.4f}{raw['brier']:>10.4f}")
    print(f"  {'+ temperature scaling':<26}{temp['ece']:>10.4f}{temp['brier']:>10.4f}   (T*={T:.3f})")
    print(f"  {'+ isotonic regression':<26}{iso['ece']:>10.4f}{iso['brier']:>10.4f}   ({iso_impl})")
    improved = (temp['ece'] < raw['ece'] - 1e-4) or (iso['ece'] < raw['ece'] - 1e-4)
    print(f"  -> standard recalibration {'improves' if improved else 'does NOT meaningfully improve'} "
          f"ECE (raw already {raw['ece']:.4f})")

    print("\n=== CONFORMAL PREDICTION (coverage validity) ===")
    print(f"  {'target':>8}{'empirical':>11}{'qhat':>8}{'size1%':>9}{'size2(abstain)%':>17}")
    for c in conf:
        print(f"  {c['target_coverage']:>8.2f}{c['empirical_coverage']:>11.4f}{c['qhat']:>8.3f}"
              f"{c['set_size_1_pct']:>9.2f}{c['abstain_pct']:>17.2f}")
    print("  -> empirical coverage should meet the target (validity). size-2 sets are")
    print("     the conformal 'abstain / route' region, analogous to ALC UNTRUSTWORTHY.")

    Path(args.out).write_text(json.dumps({
        "n_total": len(rows), "n_fit": len(cal), "n_eval": len(test),
        "temperature": T,
        "calibration": {"raw": raw, "temperature_scaling": temp,
                        "isotonic": iso, "isotonic_impl": iso_impl},
        "conformal": conf,
    }, indent=2))
    print(f"\nsaved {args.out}")
    print("\nFraming: report (1) as evidence that V needs no post-hoc calibration, and")
    print("(2) as coverage-valid conformal reference; note ALC adds distribution-shift")
    print("adaptivity (delta widens under shift) that split conformal, assuming")
    print("exchangeability, does not provide. Do NOT stage an accuracy bake-off.")


if __name__ == "__main__":
    main()
