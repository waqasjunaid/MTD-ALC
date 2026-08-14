# -*- coding: utf-8 -*-
"""
regen_armB.py

Generates the Arm B (CWE-only ablation baseline) repair patch for the
same 548 samples as Arm A, per the ablation design:

  Arm A (diagnosis-guided, your real method): prompt = <f, c*, eps*, k*>
    + dangerous constructs + taint paths + risky lines
  Arm B (CWE-only baseline, mimics Pearce et al./Fu et al.): prompt = <f, c*>
    only -- function and dominant CWE, nothing else from diagnosis.

KEY DESIGN DECISION (much faster than Arm A):
  Arm A's regen_armA.py re-ran MTD -> ALC -> Diagnosis -> Repair as four
  subprocess calls per sample, because it needed the diagnosis module's
  full output (constructs, taint, risky lines) for the prompt.
  Arm B only needs the dominant CWE label, which is ALREADY CACHED in
  diagnosis_records.jsonl (the 'dominant_cwe' field) from your original
  run. So this script skips MTD/ALC/Diagnosis entirely and calls the
  LLM directly, once per sample -- no subprocess pipeline, no waiting
  on the other three modules. This should take roughly 1/3-1/4 of
  Arm A's wall-clock time.

  This is a deliberate scope reduction, not a hidden shortcut: only the
  Stage-1 fix patch is needed for correctness-metric comparison against
  reference_fix (exact-match/CodeBLEU/construct-removal do not use
  Stage 2/3 outputs), so Arm B does not run Stages 2-3 at all. Because
  Stage 2/3 in Arm A are conditioned on Stage 1's output and do not feed
  back into it, this does not bias the diag-vs-cwe_only comparison,
  which is scored on Stage-1 patches for both arms.

REUSES, UNCHANGED, FROM repair/run_repair.py:
  - OLLAMA_HOST, MODEL_NAME, TEMPERATURE, MAX_TOKENS (same model/config
    as Arm A -- critical for a valid ablation)
  - call_llm()      (identical LLM call path)
  - _strip_fences() (identical output cleaning)
  - _CWE_MITIGATIONS / _DEFAULT_MIT (kept in Arm B's prompt too, since
    this is a static catalogue keyed by CWE label, not diagnosis-module
    output -- arguably part of "knowing the CWE" rather than part of
    the diagnosis-guided context being ablated away. If you'd rather
    strip this too for a maximally minimal baseline, set
    INCLUDE_MITIGATION_CATALOGUE = False below.)

Usage:
    python regen_armB.py \
        --records diagnosis_records.jsonl \
        --archive patches_armB \
        --root . \
        [--limit 5]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repair"))

from run_repair import (
    call_llm, _strip_fences, _CWE_MITIGATIONS, _DEFAULT_MIT,
    MODEL_NAME, OLLAMA_HOST, TEMPERATURE, MAX_TOKENS,
)

INCLUDE_MITIGATION_CATALOGUE = True  # see note above


def guess_func_name(vulnerable_func: str) -> str:
    """Best-effort function name extraction for the prompt's log/header
    context only -- does not affect scoring."""
    m = re.search(r"\b([A-Za-z_]\w*)\s*\(", vulnerable_func)
    return m.group(1) if m else "unknown_function"


def build_prompt_arm_b(func_code: str, func_name: str, dominant_cwe: str):
    mitigation = _CWE_MITIGATIONS.get(dominant_cwe, _DEFAULT_MIT)

    system_prompt = (
        "You are a senior C security engineer specialising in vulnerability remediation. "
        "You will be given a C function and its associated CWE category. "
        "Your task is to produce a MINIMAL, CORRECT patch.\n\n"
        "STRICT RULES:\n"
        "1. Fix ONLY the identified vulnerability -- do not refactor unrelated code.\n"
        "2. Preserve the original function signature EXACTLY.\n"
        "3. Add an inline comment on EVERY changed line: /* FIX [CWE-xxx]: reason */\n"
        "4. Add a comment block at the very top listing all changes made.\n"
        "5. If no real vulnerability exists, return the original with a comment explaining why.\n"
        "6. Output ONLY the patched C function -- no markdown, no explanation outside the code."
    )

    mitigation_block = (
        f"CWE MITIGATION GUIDANCE:\n  {mitigation}\n\n"
        if INCLUDE_MITIGATION_CATALOGUE else ""
    )

    user_prompt = (
        f"DIAGNOSIS REPORT:\n"
        f"  Function name:   {func_name}\n"
        f"  Dominant CWE:    {dominant_cwe}\n\n"
        f"{mitigation_block}"
        f"ORIGINAL FUNCTION CODE:\n"
        f"{func_code}\n\n"
        f"OUTPUT THE PATCHED FUNCTION NOW:"
    )
    return system_prompt, user_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="diagnosis_records.jsonl")
    ap.add_argument("--archive", required=True, help="output dir, e.g. patches_armB")
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"Model config (must match Arm A): {MODEL_NAME} @ {OLLAMA_HOST}  "
          f"temp={TEMPERATURE}  max_tokens={MAX_TOKENS}")
    print(f"Include CWE mitigation catalogue: {INCLUDE_MITIGATION_CATALOGUE}")

    archive = Path(args.archive)
    archive.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]

    done = failed = skipped = 0
    fail_log = open("armB_failures.txt", "a", encoding="utf-8")

    for rec in records:
        sid = str(rec["id"])
        dest = archive / f"{sid}.c"
        if dest.exists():
            skipped += 1
            continue

        vulnerable_func = rec.get("vulnerable_func", "")
        dominant_cwe = rec.get("dominant_cwe") or "unknown"
        if not vulnerable_func:
            print(f"[{sid}] no vulnerable_func in record")
            failed += 1
            fail_log.write(f"{sid}\tno_source\n")
            continue

        func_name = guess_func_name(vulnerable_func)
        system_prompt, user_prompt = build_prompt_arm_b(
            vulnerable_func, func_name, dominant_cwe
        )

        patched = call_llm(system_prompt, user_prompt, "ArmB-Stage1-Fix")
        patched = _strip_fences(patched)
        patch_generated = not patched.startswith("[LLM ERROR")

        if patch_generated and patched.strip():
            dest.write_text(patched, encoding="utf-8")
            done += 1
            print(f"[{sid}] OK real patch -> {dest}  ({done} done)")
        else:
            failed += 1
            fail_log.write(f"{sid}\tllm_error\n")
            print(f"[{sid}] FAILED: {patched[:200]}")

    fail_log.close()
    print(f"\nDONE. real={done}  failed={failed}  skipped={skipped}  "
          f"archived_total={len(list(archive.glob('*.c')))}")
    print("Failures logged to armB_failures.txt")


if __name__ == "__main__":
    main()
