#!/usr/bin/env python3
"""
score_repair_llmjudge.py

Reference-free LLM-as-judge scoring of repair quality, run against a LOCAL
Ollama model (your Qwen 2.5), following the LLM4CVE precedent (ref [37]).

The judge sees ONLY: the vulnerable function, the (optional) diagnosed CWE, and
a candidate patch. It never sees the ground-truth fix -- we are measuring whether
the patch addresses the vulnerability, NOT whether it reproduces a reference
string (that is exact-match, the metric that structurally favors VulRepair).

Two modes
---------
single : score every patch in one JSONL (answers "does our repair address the vuln?")
    python score_repair_llmjudge.py --jsonl results.jsonl --out judge_diag.jsonl \
        --diag_jsonl diagnosis_records.jsonl --model qwen2.5:7b

ab     : BLIND paired comparison of two arms on shared ids (diag vs cwe_only).
         For each id the two patches are shown in randomized order as "Patch A"/
         "Patch B"; the script records which physical arm was which and undoes the
         blinding only at aggregation time.
    python score_repair_llmjudge.py --ab \
        --jsonl_a results.jsonl --jsonl_b results_armB.jsonl \
        --diag_jsonl diagnosis_records.jsonl --model qwen2.5:7b --out judge_ab.jsonl

Honest-use notes
----------------
* LLM judgment is NOT ground truth. Use --calibrate N to dump a random subset for
  human review so you can report human/LLM agreement, exactly as LLM4CVE did.
* temperature is 0 for determinism; even so, report this as an LLM-judged metric,
  paired with your independent Flawfinder result -- not as an oracle.
* Long local-LLM runs: results are appended incrementally; --resume skips finished
  ids so an interrupted run continues where it left off. Use nohup for the full set.
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

RUBRIC_KEYS = ["addresses_vulnerability", "introduces_new_problem",
               "preserves_behavior", "overall_security_score"]

JUDGE_INSTRUCTIONS = (
    "You are a strict security code reviewer for C/C++.\n"
    "You are given a function that contains a security vulnerability, optionally its "
    "CWE category, and ONE candidate patched version of that function.\n"
    "Judge ONLY whether the patch addresses the security vulnerability. Do NOT reward "
    "comments, renaming, formatting, or stylistic changes. A patch that only adds a "
    "comment or an unrelated guard does NOT address the vulnerability.\n"
    "Return STRICT JSON with exactly these keys and no prose:\n"
    '{\n'
    '  "addresses_vulnerability": 0|1|2,   // 0=not addressed, 1=partially, 2=fully addressed\n'
    '  "introduces_new_problem": 0|1|2,    // 0=introduces a serious new bug/vuln, 1=minor/uncertain, 2=none\n'
    '  "preserves_behavior": 0|1|2,        // 0=breaks intended behavior, 1=uncertain, 2=preserved\n'
    '  "overall_security_score": 1..10,    // holistic security quality of the patch\n'
    '  "one_line_reason": "short justification"\n'
    '}\n'
)


def build_prompt(vuln_func, cwe, patch):
    cwe_line = f"CWE category: {cwe}\n" if cwe else "CWE category: (not provided; infer it)\n"
    return (
        JUDGE_INSTRUCTIONS
        + "\n--- VULNERABLE FUNCTION ---\n" + vuln_func.strip()
        + "\n\n" + cwe_line
        + "\n--- CANDIDATE PATCH ---\n" + patch.strip()
        + "\n\nReturn only the JSON object."
    )


def build_ab_prompt(vuln_func, cwe, patch_a, patch_b):
    cwe_line = f"CWE category: {cwe}\n" if cwe else "CWE category: (not provided; infer it)\n"
    return (
        "You are a strict security code reviewer for C/C++.\n"
        "You are given a vulnerable function, optionally its CWE, and TWO candidate "
        "patches (A and B). Score EACH patch independently on whether it addresses the "
        "vulnerability. Do NOT reward comments or stylistic changes.\n"
        "Return STRICT JSON with this exact shape and no prose:\n"
        '{ "A": {"addresses_vulnerability":0|1|2, "introduces_new_problem":0|1|2, '
        '"preserves_behavior":0|1|2, "overall_security_score":1..10}, '
        '"B": {"addresses_vulnerability":0|1|2, "introduces_new_problem":0|1|2, '
        '"preserves_behavior":0|1|2, "overall_security_score":1..10} }\n'
        + "\n--- VULNERABLE FUNCTION ---\n" + vuln_func.strip()
        + "\n\n" + cwe_line
        + "\n--- PATCH A ---\n" + patch_a.strip()
        + "\n\n--- PATCH B ---\n" + patch_b.strip()
        + "\n\nReturn only the JSON object."
    )


def ollama_generate(prompt, model, host, timeout=180):
    """Call local Ollama /api/generate with JSON-forced output, temperature 0."""
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "format": "json", "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def parse_json_leniently(text):
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = max(parts, key=len)
        nl = text.find("\n")
        if 0 <= nl <= 6:
            text = text[nl + 1:]
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def judge_call(prompt, model, host, retries=3):
    """Call the judge with retries; return parsed dict or None."""
    for attempt in range(retries):
        try:
            raw = ollama_generate(prompt, model, host)
            obj = parse_json_leniently(raw)
            if obj is not None:
                return obj
        except Exception as e:
            if attempt == retries - 1:
                return {"_error": str(e)}
        time.sleep(1.0)
    return {"_error": "unparseable JSON after retries"}


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def clean_code(text):
    if not text:
        return ""
    s = str(text)
    if "```" in s:
        blocks = s.split("```")[1::2]
        if blocks:
            s = max(blocks, key=len)
            nl = s.find("\n")
            if 0 <= nl <= 6:
                s = s[nl + 1:]
    return s.strip()


def load_jsonl(path, key="id"):
    rows = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows[str(d.get(key))] = d
    return rows


def load_cwe_map(diag_path):
    if not diag_path:
        return {}
    m = {}
    for line in open(diag_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        m[str(d.get("id"))] = d.get("dominant_cwe") or d.get("cwe") or ""
    return m


def done_ids(out_path):
    if not Path(out_path).exists():
        return set()
    ids = set()
    for line in open(out_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(str(json.loads(line)["id"]))
        except Exception:
            pass
    return ids


def append_jsonl(out_path, rec):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def summarize_single(out_path):
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    good = [r for r in recs if "verdict" in r and isinstance(r["verdict"], dict)
            and "overall_security_score" in r["verdict"]]
    n = len(good)
    if not n:
        print("no scored records yet"); return
    def val(r, k): return r["verdict"].get(k)
    addressed = [r for r in good if (val(r, "addresses_vulnerability") or 0) >= 1]
    fully = [r for r in good if (val(r, "addresses_vulnerability") or 0) == 2]
    new_prob = [r for r in good if (val(r, "introduces_new_problem") or 2) == 0]
    scores = [val(r, "overall_security_score") for r in good
              if isinstance(val(r, "overall_security_score"), (int, float))]
    mean = sum(scores) / len(scores) if scores else 0
    errs = sum(1 for r in recs if "_error" in r.get("verdict", {}))
    print(f"\n=== LLM-JUDGE (single) : n = {n} scored ({errs} errors) ===")
    print(f"  addresses vulnerability (partial+full): {len(addressed)} ({100*len(addressed)/n:.1f}%)")
    print(f"    fully addressed:                      {len(fully)} ({100*len(fully)/n:.1f}%)")
    print(f"  introduces a new problem:               {len(new_prob)} ({100*len(new_prob)/n:.1f}%)")
    print(f"  mean overall_security_score (1-10):     {mean:.2f}")


def summarize_ab(out_path):
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    good = [r for r in recs if "arm_scores" in r]
    n = len(good)
    if not n:
        print("no scored A/B records yet"); return
    a_wins = b_wins = ties = 0
    a_addr = b_addr = 0
    da, db = [], []
    for r in good:
        sa = r["arm_scores"].get("a", {}); sb = r["arm_scores"].get("b", {})
        oa = sa.get("overall_security_score"); ob = sb.get("overall_security_score")
        if isinstance(oa, (int, float)) and isinstance(ob, (int, float)):
            da.append(oa); db.append(ob)
            if oa > ob: a_wins += 1
            elif ob > oa: b_wins += 1
            else: ties += 1
        a_addr += 1 if (sa.get("addresses_vulnerability") or 0) >= 1 else 0
        b_addr += 1 if (sb.get("addresses_vulnerability") or 0) >= 1 else 0
    # a = arm_a (diag by convention), b = arm_b (cwe_only)
    print(f"\n=== LLM-JUDGE (blind A/B) : n = {n} paired ===")
    print(f"  arm A (jsonl_a) mean score: {sum(da)/len(da):.2f}" if da else "  no scores")
    print(f"  arm B (jsonl_b) mean score: {sum(db)/len(db):.2f}" if db else "")
    print(f"  A better: {a_wins} | B better: {b_wins} | tie: {ties}")
    print(f"  addressed vuln:  A {a_addr} ({100*a_addr/n:.1f}%)  vs  B {b_addr} ({100*b_addr/n:.1f}%)")
    try:
        from scipy.stats import wilcoxon, binomtest
        nz = [x - y for x, y in zip(da, db) if x != y]
        if len(nz) >= 6:
            _, p = wilcoxon(nz)
            print(f"  Wilcoxon on paired overall scores: p = {p:.4g}")
        disc = a_wins + b_wins
        if disc:
            p2 = binomtest(min(a_wins, b_wins), disc, 0.5).pvalue
            print(f"  sign test on wins (A={a_wins}, B={b_wins}): p = {p2:.4g}")
    except Exception as e:
        print(f"  (stats unavailable: {e})")


# ----------------------------------------------------------------------------
def run_single(args):
    rows = load_jsonl(args.jsonl)
    cwe = load_cwe_map(args.diag_jsonl)
    finished = done_ids(args.out) if args.resume else set()
    if not args.resume and Path(args.out).exists():
        Path(args.out).unlink()
    ids = list(rows.keys())
    if args.max_samples:
        ids = ids[: args.max_samples]
    todo = [i for i in ids if i not in finished]
    print(f"single mode: {len(todo)} to score ({len(finished)} already done), model={args.model}")
    for k, sid in enumerate(todo, 1):
        d = rows[sid]
        vuln = clean_code(d.get(args.original_col))
        patch = clean_code(d.get(args.repaired_col))
        if not vuln or not patch:
            append_jsonl(args.out, {"id": sid, "verdict": {"_error": "empty vuln or patch"}})
            continue
        verdict = judge_call(build_prompt(vuln, cwe.get(sid, ""), patch), args.model, args.host)
        append_jsonl(args.out, {"id": sid, "cwe": cwe.get(sid, ""), "verdict": verdict})
        if k % 10 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}", flush=True)
    summarize_single(args.out)


def run_ab(args):
    a = load_jsonl(args.jsonl_a)
    b = load_jsonl(args.jsonl_b)
    cwe = load_cwe_map(args.diag_jsonl)
    shared = [i for i in a if i in b]
    finished = done_ids(args.out) if args.resume else set()
    if not args.resume and Path(args.out).exists():
        Path(args.out).unlink()
    if args.max_samples:
        shared = shared[: args.max_samples]
    todo = [i for i in shared if i not in finished]
    rng = random.Random(args.seed)
    print(f"A/B mode: {len(todo)} pairs to score ({len(finished)} done), "
          f"A={args.jsonl_a}, B={args.jsonl_b}, model={args.model}")
    for k, sid in enumerate(todo, 1):
        vuln = clean_code(a[sid].get(args.original_col)) or clean_code(b[sid].get(args.original_col))
        pa = clean_code(a[sid].get(args.repaired_col))
        pb = clean_code(b[sid].get(args.repaired_col))
        if not vuln or not pa or not pb:
            append_jsonl(args.out, {"id": sid, "arm_scores": {}, "note": "empty input"})
            continue
        # blind: randomly assign which physical arm is shown first
        a_first = rng.random() < 0.5
        first, second = (pa, pb) if a_first else (pb, pa)
        raw = judge_call(build_ab_prompt(vuln, cwe.get(sid, ""), first, second),
                         args.model, args.host)
        # de-blind: map shown A/B back to physical arm a/b
        shown = {"A": raw.get("A", {}), "B": raw.get("B", {})} if isinstance(raw, dict) else {}
        if a_first:
            arm_scores = {"a": shown.get("A", {}), "b": shown.get("B", {})}
        else:
            arm_scores = {"a": shown.get("B", {}), "b": shown.get("A", {})}
        append_jsonl(args.out, {"id": sid, "cwe": cwe.get(sid, ""),
                                "a_shown_first": a_first, "arm_scores": arm_scores,
                                "raw_error": raw.get("_error") if isinstance(raw, dict) else None})
        if k % 10 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}", flush=True)
    summarize_ab(args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="blind A/B mode (needs --jsonl_a/--jsonl_b)")
    ap.add_argument("--jsonl", help="single mode input")
    ap.add_argument("--jsonl_a"); ap.add_argument("--jsonl_b")
    ap.add_argument("--diag_jsonl", help="diagnosis_records.jsonl to pull dominant_cwe by id")
    ap.add_argument("--original_col", default="vulnerable_func")
    ap.add_argument("--repaired_col", default="generated_patch")
    ap.add_argument("--model", default="qwen2.5", help="Ollama model tag, e.g. qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--resume", action="store_true", help="skip ids already in --out")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="judge_scores.jsonl")
    ap.add_argument("--summary_only", action="store_true", help="just re-print aggregate from --out")
    args = ap.parse_args()

    if args.summary_only:
        (summarize_ab if args.ab else summarize_single)(args.out)
        return
    if args.ab:
        if not (args.jsonl_a and args.jsonl_b):
            ap.error("--ab needs --jsonl_a and --jsonl_b")
        run_ab(args)
    else:
        if not args.jsonl:
            ap.error("single mode needs --jsonl")
        run_single(args)


if __name__ == "__main__":
    main()
