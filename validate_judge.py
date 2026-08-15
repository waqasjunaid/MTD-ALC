#!/usr/bin/env python3
"""
validate_judge.py

Validate the LLM-as-judge WITHOUT human experts, using controls you already have.

The judge is only trustworthy if it scores known-good fixes high and known-bad
"fixes" low. We test exactly that, reusing the SAME judge (same model, prompt,
temperature) as score_repair_llmjudge.py so this validates the real configuration:

  positive control : the ground-truth reference_fix  (should score HIGH)
  negative control : the unmodified vulnerable_func   (should score LOW --
                     it is presented as a "patch" but changes nothing)

Headline metric: AUC of the judge's overall_security_score at separating the
positive from the negative controls. A high AUC (say > 0.85) means the judge
discriminates correct from incorrect fixes on data with known labels -- the
closest thing to human calibration available here. We also report mean scores per
group and a paired Wilcoxon test, and (if you point at your already-scored
generated patches) place them on the same scale.

Usage:
  python validate_judge.py --jsonl results.jsonl --model llama3.1:70b \
      --max_samples 150 --out_ref judge_ref.jsonl --out_vuln judge_vuln.jsonl \
      --generated_scores judge_diag.jsonl

  # re-print the comparison later without re-scoring:
  python validate_judge.py --summary_only \
      --out_ref judge_ref.jsonl --out_vuln judge_vuln.jsonl \
      --generated_scores judge_diag.jsonl

Long 70B runs: results append incrementally and --resume skips finished ids.
Use nohup for large --max_samples. 150 per control is plenty for a tight AUC.
"""

import argparse
import re
import sys
from pathlib import Path

# reuse the exact judge harness so this validates the real configuration
import score_repair_llmjudge as J


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def overall(rec):
    v = rec.get("verdict", {})
    s = v.get("overall_security_score")
    return s if isinstance(s, (int, float)) else None


def addressed(rec):
    v = rec.get("verdict", {})
    a = v.get("addresses_vulnerability")
    return a if isinstance(a, (int, float)) else None


def score_group(rows, ids, patch_key, out_path, cwe_map, args, group_name):
    """Score one control group: for each id, judge (vulnerable_func, <patch_key>)."""
    finished = J.done_ids(out_path) if args.resume else set()
    if not args.resume and Path(out_path).exists():
        Path(out_path).unlink()
    todo = [i for i in ids if i not in finished]
    print(f"[{group_name}] {len(todo)} to score ({len(finished)} done) -> {out_path}")
    for k, sid in enumerate(todo, 1):
        d = rows[sid]
        vuln = J.clean_code(d.get(args.original_col))
        patch = J.clean_code(d.get(patch_key))
        if not vuln or not patch:
            J.append_jsonl(out_path, {"id": sid, "verdict": {"_error": "empty"}})
            continue
        verdict = J.judge_call(J.build_prompt(vuln, cwe_map.get(sid, ""), patch),
                               args.model, args.host)
        J.append_jsonl(out_path, {"id": sid, "verdict": verdict})
        if k % 10 == 0 or k == len(todo):
            print(f"  [{group_name}] {k}/{len(todo)}", flush=True)


def auc(pos, neg):
    """Rank-based AUC = P(pos > neg) + 0.5 P(tie). Equivalent to Mann-Whitney U/(n1 n2)."""
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return None
    allv = sorted([(v, 0) for v in pos] + [(v, 1) for v in neg])
    # assign average ranks
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1
    r_pos = sum(r for r, (_, lbl) in zip(ranks, allv) if lbl == 0)
    u = r_pos - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


def load_scores(path):
    if not path or not Path(path).exists():
        return []
    return [__import__("json").loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def summarize(args):
    import statistics as st
    ref = load_scores(args.out_ref)
    vul = load_scores(args.out_vuln)
    gen = load_scores(args.generated_scores) if args.generated_scores else []

    def scores(recs):
        return [overall(r) for r in recs if overall(r) is not None]

    rs, vs, gs = scores(ref), scores(vul), scores(gen)
    if not rs or not vs:
        print("Need both control groups scored first.")
        return

    def addr_rate(recs):
        a = [addressed(r) for r in recs if addressed(r) is not None]
        return (sum(1 for x in a if x >= 1) / len(a)) if a else 0.0

    print("\n=== JUDGE VALIDATION (controls with known labels) ===")
    print(f"  positive control  (reference fixes):        n={len(rs)}  "
          f"mean={st.mean(rs):.2f}  addressed={100*addr_rate(ref):.1f}%")
    print(f"  negative control  (unmodified vulnerable):  n={len(vs)}  "
          f"mean={st.mean(vs):.2f}  addressed={100*addr_rate(vul):.1f}%")
    if gs:
        print(f"  your generated patches (for reference):     n={len(gs)}  "
              f"mean={st.mean(gs):.2f}  addressed={100*addr_rate(gen):.1f}%")

    a = auc(rs, vs)
    print(f"\n  AUC (judge score separates good from bad fixes): {a:.3f}")
    sep = "clean" if a is not None and a >= 0.85 else ("moderate" if a and a >= 0.7 else "weak")
    print(f"  -> separation is {sep.upper()} "
          f"(>=0.85 clean, 0.70-0.85 moderate, <0.70 weak)")

    # paired Wilcoxon on ids present in both controls
    try:
        from scipy.stats import wilcoxon, mannwhitneyu
        rmap = {r["id"]: overall(r) for r in ref if overall(r) is not None}
        vmap = {r["id"]: overall(r) for r in vul if overall(r) is not None}
        shared = [i for i in rmap if i in vmap]
        deltas = [rmap[i] - vmap[i] for i in shared if rmap[i] != vmap[i]]
        if len(deltas) >= 6:
            _, p = wilcoxon(deltas, alternative="greater")  # H1: reference > vulnerable
            print(f"  paired Wilcoxon (reference > vulnerable): p = {p:.4g} "
                  f"(n_pairs={len(shared)}, non-tied={len(deltas)})")
        _, pu = mannwhitneyu(rs, vs, alternative="greater")
        print(f"  Mann-Whitney U (reference > vulnerable):  p = {pu:.4g}")
    except Exception as e:
        print(f"  (stats unavailable: {e})")

    print("\n  interpretation: if the judge scores known-good fixes well above")
    print("  unmodified-vulnerable code, it discriminates correct from incorrect")
    print("  repairs on labeled data -- validating the metric without human review.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="results.jsonl (needs vulnerable_func + reference_fix)")
    ap.add_argument("--original_col", default="vulnerable_func")
    ap.add_argument("--reference_col", default="reference_fix")
    ap.add_argument("--diag_jsonl", help="optional: dominant_cwe by id")
    ap.add_argument("--generated_scores", help="your already-scored patches (judge_diag.jsonl) for context")
    ap.add_argument("--model", default="llama3.1:70b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max_samples", type=int, default=150,
                    help="samples PER control group (0 = all). 150 gives a tight AUC.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out_ref", default="judge_ref.jsonl")
    ap.add_argument("--out_vuln", default="judge_vuln.jsonl")
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        summarize(args)
        return
    if not args.jsonl:
        ap.error("need --jsonl to score controls (or use --summary_only)")
    if not Path(args.jsonl).exists():
        sys.exit(f"ERROR: --jsonl not found: {args.jsonl}")

    rows = J.load_jsonl(args.jsonl)
    cwe_map = J.load_cwe_map(args.diag_jsonl) if args.diag_jsonl else {}

    # drop degenerate samples where the reference fix == the vulnerable code
    # (nothing to distinguish; would pollute both controls)
    usable = []
    skipped = 0
    for sid, d in rows.items():
        v = norm(J.clean_code(d.get(args.original_col)))
        r = norm(J.clean_code(d.get(args.reference_col)))
        if not v or not r:
            skipped += 1
            continue
        if v == r:
            skipped += 1
            continue
        usable.append(sid)
    if skipped:
        print(f"note: skipped {skipped} degenerate/empty samples "
              f"(reference == vulnerable or missing).")
    ids = usable[: args.max_samples] if args.max_samples else usable
    print(f"scoring {len(ids)} ids per control group, model={args.model}")

    # negative control: the vulnerable function itself is the "patch"
    score_group(rows, ids, args.original_col, args.out_vuln, cwe_map, args, "neg/vulnerable")
    # positive control: the reference fix is the "patch"
    score_group(rows, ids, args.reference_col, args.out_ref, cwe_map, args, "pos/reference")

    summarize(args)


if __name__ == "__main__":
    main()
