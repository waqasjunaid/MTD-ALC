#!/usr/bin/env python3
"""
verify_patches.py

Closes the "open-loop repair" gap flagged in the very first methodology
review of this project: re-runs the trained MTD (Task 1-4 + ensemble V
score) on each repaired function, so we can report whether the model's
OWN detector still thinks the patched code is vulnerable.

Design notes (why it's done this way):

  - V_before for every sample is already in bigvul_results.csv from your
    original full-dataset run -- no need to re-score the original code.

  - For V_after, the repaired function has no ground-truth flaw-line
    labels (nobody has hand-annotated an LLM-generated patch), so
    suspicious_lines/line_map can't use the "ground_truth" strategy.
    codepreprocesing/suspicious_line_mapper.py already has a built-in
    fallback for exactly this situation: passing flaw_lines=None
    triggers strategy="heuristic" automatically -- the same fallback
    your pipeline already uses for MegaVul. This is not a new hack;
    it's existing, intended behavior of your own code.

  - task1.run() prefers func['code'] over reading source_file from disk,
    so we pass the patch text directly -- but we still also write it to
    disk (verify_sources/{id}.c) for debugging/audit purposes.

Run this from your project ROOT (same directory as mtd/, alc/,
codepreprocesing/, repair/, diagnosis/), so the imports resolve exactly
like run_mtd.py's do.

Usage:
    python verify_patches.py \
        --results results.jsonl \
        --bigvul-results bigvul_results.csv \
        --out verify_results.jsonl \
        --srcdir verify_sources
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mtd"))
sys.path.insert(0, str(ROOT / "mtd" / "ml"))

from codepreprocesing.function_extractor import extract_from_string
from codepreprocesing.suspicious_line_mapper import map_suspicious_lines

import task1_vulnerability_classification as task1
import task2_line_localization as task2
import task3_syntax_risk_prediction as task3
import task4_dependency_propagation_risk as task4
from feature_extractor import extract as extract_features
from infer import score_ensemble, get_opt_threshold, ModelNotTrainedError


def load_before_scores(path):
    """sample_id -> {V_score, mtd_verdict} from the original full run.
    NOTE: kept for reference/reporting only -- these were computed under
    'ground_truth' strategy (confidence=1.0 by construction), which is
    NOT comparable to the 'heuristic' strategy used for V_after. See
    score_original_heuristic() for the fair, apples-to-apples score."""
    before = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            before[str(row["sample_id"])] = {
                "V_before_groundtruth": float(row["V_score"]),
                "verdict_before_groundtruth": row["mtd_verdict"],
                "strategy_before": row["strategy"],
            }
    return before


def score_original_heuristic(vulnerable_func: str, sample_id: str):
    """Re-score the ORIGINAL vulnerable function under 'heuristic' strategy
    (flaw_lines=None), matching exactly how V_after is computed, so the
    strategy variable is controlled for and the comparison is fair."""
    func = pick_target_function(vulnerable_func, sample_id)
    src_path = Path("verify_sources_before") / f"{sample_id}.c"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(func["code"], encoding="utf-8")
    func["source_file"] = str(src_path)
    return score_patch(func, str(src_path))


import re as _re

# Strips the LLM's OWN injected commentary, per the exact format required
# by repair/run_repair.py's system prompt:
#   "3. Add an inline comment on EVERY changed line: /* FIX [CWE-xxx]: reason */"
#   "4. Add a comment block at the very top listing all changes made."
# These annotations are text your MTD models never saw in training (BigVul
# has no such comments), and diagnostics showed they can drag the ensemble
# V score toward "benign" even when a severe construct (system(), execve(),
# unchecked memmove()) is still literally present in the code. Scoring
# should compare code to code, not code-plus-self-narration.
_INLINE_FIX_COMMENT_RE = _re.compile(r"/\*\s*FIX\b.*?\*/", _re.IGNORECASE | _re.DOTALL)
_CHANGES_BLOCK_RE = _re.compile(
    r"/\*\s*(CHANGES MADE|Changes made|Summary of changes)\s*:?.*?\*/",
    _re.IGNORECASE | _re.DOTALL,
)


def strip_llm_annotations(code: str) -> str:
    code = _CHANGES_BLOCK_RE.sub("", code)
    code = _INLINE_FIX_COMMENT_RE.sub("", code)
    return code


def pick_target_function(patch_text: str, sample_id: str):
    """Extract functions from the patch text and pick the main one (the
    largest by code length -- the header comment isn't a function, and
    any tiny helper snippets shouldn't outrank the actual repaired body)."""
    funcs = extract_from_string(patch_text, source_id=sample_id)
    if not funcs:
        # Fallback: treat the whole text as one function so scoring can
        # still run rather than silently dropping the sample.
        return {
            "id": sample_id,
            "name": "unknown",
            "code": patch_text,
            "start_line": 1,
            "end_line": len(patch_text.splitlines()),
            "param_count": 0,
            "line_count": len(patch_text.splitlines()),
            "source_file": None,
        }
    return max(funcs, key=lambda f: len(f.get("code", "")))


def score_patch(func: dict, source_file: str):
    line_map = map_suspicious_lines(func, flaw_lines=None)  # -> heuristic
    suspicious_lines = line_map["suspicious_line_numbers"]

    r1 = task1.run(source_file, suspicious_lines, func, line_map)
    r2 = task2.run(source_file, suspicious_lines, func, line_map)
    r3 = task3.run(source_file, suspicious_lines, func, line_map)
    r4 = task4.run(source_file, suspicious_lines, func, line_map)

    features = extract_features(r1, r2, r3, r4, func, line_map, suspicious_lines)
    V = round(float(score_ensemble(features)), 4)
    return V, line_map["strategy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.jsonl (has generated_patch)")
    ap.add_argument("--bigvul-results", required=True, help="bigvul_results.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--srcdir", default="verify_sources")
    args = ap.parse_args()

    try:
        opt_threshold = get_opt_threshold()
    except ModelNotTrainedError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    before = load_before_scores(args.bigvul_results)
    srcdir = Path(args.srcdir)
    srcdir.mkdir(parents=True, exist_ok=True)

    n = 0
    n_no_before = 0
    n_flipped = 0        # fair (heuristic-vs-heuristic): was>=thr, now<thr
    n_still_vuln = 0      # fair: was>=thr, still>=thr
    n_error = 0
    deltas = []

    with open(args.results) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec["id"])
            n += 1

            b = before.get(sid)
            if b is None:
                n_no_before += 1
                continue

            try:
                clean_patch = strip_llm_annotations(rec["generated_patch"])
                func = pick_target_function(clean_patch, sid)
                src_path = srcdir / f"{sid}.c"
                src_path.write_text(func["code"], encoding="utf-8")
                func["source_file"] = str(src_path)

                V_after, strategy_after = score_patch(func, str(src_path))
                V_before_fair, strategy_before_fair = score_original_heuristic(
                    rec["vulnerable_func"], sid
                )
            except Exception as e:
                n_error += 1
                print(f"[{sid}] verification failed: {e}")
                continue

            was_vuln = V_before_fair >= opt_threshold
            now_vuln = V_after >= opt_threshold
            delta = V_before_fair - V_after
            deltas.append(delta)

            if was_vuln and not now_vuln:
                n_flipped += 1
            elif was_vuln and now_vuln:
                n_still_vuln += 1

            out_rec = {
                "id": sid,
                "V_before_groundtruth": b["V_before_groundtruth"],
                "V_before_heuristic": V_before_fair,
                "V_after": V_after,
                "delta_fair": round(delta, 4),
                "was_vulnerable_fair": was_vuln,
                "still_vulnerable_fair": now_vuln,
            }
            fout.write(json.dumps(out_rec) + "\n")

    print(f"\ntotal patches:                 {n}")
    print(f"skipped (no before-score):    {n_no_before}")
    print(f"skipped (scoring error):      {n_error}")
    scored = n - n_no_before - n_error
    print(f"successfully scored:          {scored}")
    if scored:
        print(f"opt_threshold:                {opt_threshold}")
        print(f"[FAIR, heuristic-vs-heuristic]")
        print(f"self-verified fixed "
              f"(was>=thr, now<thr):         {n_flipped}/{scored} "
              f"({100*n_flipped/scored:.1f}%)")
        print(f"still flagged vulnerable:     {n_still_vuln}/{scored} "
              f"({100*n_still_vuln/scored:.1f}%)")
        print(f"mean V delta (before-after):  {sum(deltas)/len(deltas):.4f}")
    print(f"\nfull per-sample results written to {args.out}")


if __name__ == "__main__":
    main()