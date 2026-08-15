#!/usr/bin/env python3
"""
score_repair_redetect.py

Re-detection metric for repair improvement (no external static analyzer).

For each sample in results.jsonl it runs YOUR MTD detector on the original
vulnerable function and on the generated patch, both under the SAME heuristic
suspicious-line mapping (a fresh patch has no CVE flaw-line annotation, so
ground_truth mapping is unavailable; scoring the original under heuristic too
keeps the only difference the code itself, not the strategy). It then reports,
across ALL samples:

  * predicted-VULNERABLE rate before vs. after repair,
  * of functions MTD flags before repair, the fraction that flip to
    NON_VULNERABLE after ("repair clears the detector" rate) + Wilson CI,
  * mean dV and a Wilcoxon signed-rank test on the per-sample score drop,
  * a McNemar exact test on the paired flip (regressions reported too).

HONEST FRAMING (printed in output): this is a self-consistency measure -- your
own model judging your own repairs. It has full coverage (all 548, no analyzer
parse wall), which the Flawfinder/static-analysis route lacked, but it must be
paired with the independent Flawfinder result. It is NOT an independent oracle.

Run a 3-sample self-check FIRST (default). Only pass --full after it looks right.

Usage:
  # self-check (default): preprocess + score 3 samples, print signatures, stop
  python score_repair_redetect.py --jsonl results.jsonl

  # full run over all samples
  python score_repair_redetect.py --jsonl results.jsonl --full --out redetect.json
"""

import argparse
import json
import sys
import tempfile
import inspect
from pathlib import Path

# ---- wire up your project's import paths (edit ROOT if this script is moved) ----
ROOT = Path(__file__).resolve().parent
# The script is expected to sit at the project root alongside mtd/ and codepreprocesing/.
# If you run it from elsewhere, pass --root.
def add_paths(root):
    for p in [root, root / "mtd", root / "mtd" / "ml", root / "codepreprocesing"]:
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def load_mtd():
    """Import the MTD tasks + ML scoring, matching run_mtd.py's usage."""
    import task1_vulnerability_classification as task1
    import task2_line_localization as task2
    import task3_syntax_risk_prediction as task3
    import task4_dependency_propagation_risk as task4
    from feature_extractor import extract as extract_features
    from infer import score_ensemble, get_opt_threshold
    return {
        "tasks": (task1, task2, task3, task4),
        "extract": extract_features,
        "score": score_ensemble,
        "threshold": get_opt_threshold(),
    }


def load_preprocess():
    """Import the per-row preprocessing worker from codepreprocesing/preprocess.py."""
    import preprocess as pp
    fn = getattr(pp, "_process_row", None)
    return pp, fn


def _t2score(r2):
    """Mirror run_mtd.py's Task-2 scalarization (localization density)."""
    if isinstance(r2, dict):
        for k in ("score", "overall_localization_risk", "risk", "localization_score"):
            if k in r2:
                try:
                    return float(r2[k])
                except (TypeError, ValueError):
                    pass
    return 0.0


def score_one(code, sample_id, dataset, mtd, ppfn, tmpdir, strategy="heuristic"):
    """Preprocess a raw function under heuristic mapping, then run MTD -> V, verdict.
    Returns (V, verdict_bool, note) where note is set if scoring failed."""
    if not code or not code.strip():
        return None, None, "empty code"

    src_path = Path(tmpdir) / f"{sample_id}.c"
    src_path.write_text(code, encoding="utf-8", errors="replace")

    # Build a dataset-row shape the preprocessing worker expects. _process_row in
    # preprocess.py takes a raw row and returns:
    #   {id, label, func, source_file, line_map, suspicious_line_numbers}
    # We force heuristic mapping (no ground-truth flaw lines for a patch).
    row = {
        "id": sample_id,
        "func_before": code,
        "func": code,
        "target": code,
        "label": 1,
        "dataset": dataset,
    }
    try:
        # real signature: _process_row(row, dataset, flaw_lines, strategy)
        # force heuristic mapping with no ground-truth flaw lines (a patch has none)
        rec = ppfn(row, dataset, [], strategy)
    except Exception as e:
        return None, None, f"preprocess failed: {e}"

    if not rec:
        return None, None, "preprocess returned None"

    func = rec.get("func", {})
    line_map = rec.get("line_map", {})
    suspicious = rec.get("suspicious_line_numbers", rec.get("suspicious_lines", []))
    source_file = rec.get("source_file") or str(src_path)

    task1, task2, task3, task4 = mtd["tasks"]
    try:
        r1 = task1.run(source_file, suspicious, func, line_map)
        r2 = task2.run(source_file, suspicious, func, line_map)
        r3 = task3.run(source_file, suspicious, func, line_map)
        r4 = task4.run(source_file, suspicious, func, line_map)
        features = mtd["extract"](r1, r2, r3, r4, func, line_map, suspicious)
        V = float(mtd["score"](features))
    except Exception as e:
        return None, None, f"MTD scoring failed: {e}"

    verdict = V >= mtd["threshold"]
    return round(V, 4), bool(verdict), None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return {"discordant_b": b, "discordant_c": c, "p_value": None, "note": "no discordant pairs"}
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    except Exception:
        from math import comb
        k = min(b, c)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"discordant_b": b, "discordant_c": c, "p_value": float(p)}


def wilcoxon_drop(deltas):
    nz = [d for d in deltas if d != 0]
    if len(nz) < 6:
        return {"n_nonzero": len(nz), "p_value": None, "note": "too few non-zero deltas"}
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(nz, alternative="less")  # H1: V_after - V_before < 0
        return {"n_nonzero": len(nz), "statistic": float(stat), "p_value": float(p),
                "alternative": "V decreases after repair"}
    except Exception as e:
        return {"n_nonzero": len(nz), "p_value": None, "note": f"scipy: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--root", default=str(ROOT),
                    help="project root containing mtd/ and codepreprocesing/ (default: script dir)")
    ap.add_argument("--original_col", default="vulnerable_func")
    ap.add_argument("--repaired_col", default="generated_patch")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--dataset", default="bigvul")
    ap.add_argument("--full", action="store_true", help="run all samples (default: 3-sample self-check)")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--out", default="redetect_scores.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    add_paths(root)

    if not Path(args.jsonl).exists():
        sys.exit(f"ERROR: --jsonl not found: {args.jsonl}")

    try:
        mtd = load_mtd()
    except Exception as e:
        sys.exit(f"ERROR importing MTD (check --root={root}): {e}")
    try:
        pp, ppfn = load_preprocess()
    except Exception as e:
        sys.exit(f"ERROR importing preprocess (check --root={root}): {e}")
    if ppfn is None:
        sys.exit("ERROR: could not find _process_row in codepreprocesing/preprocess.py; "
                 "tell me the actual function name and I'll adjust.")

    print(f"threshold theta* = {mtd['threshold']}")
    try:
        print(f"_process_row signature: {inspect.signature(ppfn)}")
    except (TypeError, ValueError):
        pass

    samples = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    if not args.full:
        samples = samples[:3]
        print("\n*** SELF-CHECK MODE (3 samples). Add --full for the real run. ***\n")
    elif args.max_samples:
        samples = samples[: args.max_samples]

    records, deltas = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, d in enumerate(samples, 1):
            sid = str(d.get(args.id_col) or f"S{i}")
            vb, verb_b, nb = score_one(d.get(args.original_col), f"{sid}_orig",
                                       args.dataset, mtd, ppfn, tmp)
            va, verb_a, na = score_one(d.get(args.repaired_col), f"{sid}_rep",
                                       args.dataset, mtd, ppfn, tmp)
            rec = {"id": sid, "V_before": vb, "V_after": va,
                   "vuln_before": verb_b, "vuln_after": verb_a,
                   "note_before": nb, "note_after": na}
            records.append(rec)
            if vb is not None and va is not None:
                deltas.append(va - vb)
            if not args.full:
                print(f"  {sid}: V_before={vb} ({'VULN' if verb_b else 'clean' if verb_b is not None else nb}) "
                      f"-> V_after={va} ({'VULN' if verb_a else 'clean' if verb_a is not None else na})")
            elif i % 25 == 0 or i == len(samples):
                print(f"  processed {i}/{len(samples)}", flush=True)

    if not args.full:
        ok = sum(1 for r in records if r["V_before"] is not None and r["V_after"] is not None)
        print(f"\nself-check: {ok}/{len(records)} scored cleanly.")
        if ok < len(records):
            print("Some failed -- paste the notes above and I'll fix the preprocessing call "
                  "before you commit to the full run.")
        else:
            print("Looks good. Re-run with --full --out redetect.json for all samples.")
        return

    # ---- aggregate (full run) ----
    scored = [r for r in records if r["vuln_before"] is not None and r["vuln_after"] is not None]
    flagged_before = [r for r in scored if r["vuln_before"]]
    cleared = [r for r in flagged_before if not r["vuln_after"]]          # VULN -> clean (good)
    still_vuln = [r for r in flagged_before if r["vuln_after"]]           # VULN -> VULN
    regressions = [r for r in scored if not r["vuln_before"] and r["vuln_after"]]  # clean -> VULN (bad)

    n_scored = len(scored)
    vb_rate = sum(1 for r in scored if r["vuln_before"]) / n_scored if n_scored else 0
    va_rate = sum(1 for r in scored if r["vuln_after"]) / n_scored if n_scored else 0
    clear_lo, clear_hi = wilson_ci(len(cleared), len(flagged_before)) if flagged_before else (0, 0)
    # McNemar on paired verdict flip
    b = len(cleared)          # flagged->clean
    c = len(regressions)      # clean->flagged
    mcn = mcnemar_exact(b, c)
    wil = wilcoxon_drop(deltas)
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

    summary = {
        "n_total": len(records), "n_scored": n_scored,
        "n_failed": len(records) - n_scored,
        "vuln_rate_before": round(vb_rate, 4), "vuln_rate_after": round(va_rate, 4),
        "flagged_before": len(flagged_before),
        "cleared": len(cleared),
        "cleared_pct_of_flagged": round(100 * len(cleared) / len(flagged_before), 1) if flagged_before else 0.0,
        "cleared_wilson_ci": [round(clear_lo, 4), round(clear_hi, 4)],
        "still_vuln": len(still_vuln),
        "regressions_clean_to_vuln": len(regressions),
        "mean_delta_V": round(mean_delta, 4),
        "mcnemar": mcn, "wilcoxon": wil,
    }

    print("\n=== RE-DETECTION: MTD verdict before vs. after repair ===")
    print(f"  scored: {n_scored}/{len(records)} (failed: {summary['n_failed']})")
    print(f"  predicted-VULNERABLE rate:  before {vb_rate:.3f}  ->  after {va_rate:.3f}")
    print(f"  flagged before repair: {len(flagged_before)}")
    print(f"    cleared (VULN->clean):   {len(cleared)} "
          f"({summary['cleared_pct_of_flagged']}%)  Wilson95 {summary['cleared_wilson_ci']}")
    print(f"    still VULN:              {len(still_vuln)}")
    print(f"  regressions (clean->VULN): {len(regressions)}")
    print(f"  mean dV (after-before): {mean_delta:+.4f}")
    if wil.get("p_value") is not None:
        print(f"  Wilcoxon (H1: V drops): p = {wil['p_value']:.4g} (n_nonzero={wil['n_nonzero']})")
    if mcn.get("p_value") is not None:
        print(f"  McNemar exact on flip (b={b} cleared, c={c} regressed): p = {mcn['p_value']:.4g}")

    Path(args.out).write_text(json.dumps(
        {"framing": "SELF-CONSISTENCY: MTD scoring its own repairs under heuristic mapping; "
                    "pair with the independent Flawfinder result. Not an independent oracle.",
         "summary": summary, "records": records}, indent=2))
    print(f"\nSaved to {args.out}")
    print("\nNOTE: self-consistency measure (your detector on your patches). Report alongside")
    print("the independent Flawfinder result; do not present it as an external oracle.")


if __name__ == "__main__":
    main()