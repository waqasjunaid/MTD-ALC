#!/usr/bin/env python3
"""
generate_ablation_arms.py
-------------------------------------------------------------------
Builds results.jsonl for the repair ablation (Section V-C).

WHAT THIS SCRIPT DOES
  - Reads your per-function diagnosis records (one JSON object per function),
    which must contain everything your Repair Stage 1 needs:
        f (vulnerable source), c* (dominant CWE), eps* (error source),
        k* (outlier task), reference fix, dataset name.
  - For each function it calls the LLM TWICE:
        arm "diag"     -> prompt uses  <f, c*, eps*, k*>   (your method)
        arm "cwe_only" -> prompt uses  <f, c*>             (ablation baseline)
  - Writes results.jsonl in the format repair_ablation_metrics.py expects.

WHAT YOU MUST EDIT (two places, marked ==== EDIT ====):
  1. call_qwen(prompt) : plug in your existing Qwen 2.5 call (Ollama, etc.)
  2. load_records()    : point it at your diagnosis-output file and map the
                         field names to what your pipeline actually stores.

INPUT  : a JSONL/JSON file of diagnosis records (your data)
OUTPUT : results.jsonl  (fed to repair_ablation_metrics.py)

Run:
    python generate_ablation_arms.py diagnosis_records.jsonl results.jsonl
"""

import json
import sys


# ======================================================================
# ==== EDIT 1: your LLM call ===========================================
# Replace the body with however you already invoke Qwen 2.5 in your
# Repair module (Ollama HTTP call, subprocess, python client, ...).
# Keep temperature/seed FIXED and identical for both arms.
# ======================================================================
def call_qwen(prompt: str) -> str:
    """Return the model's Stage-1 fix text for a given prompt string."""
    # Example with Ollama's python client (uncomment & adapt):
    #
    # import ollama
    # resp = ollama.generate(
    #     model="qwen2.5",
    #     prompt=prompt,
    #     options={"temperature": 0.0, "seed": 42},
    # )
    # return resp["response"]
    #
    raise NotImplementedError(
        "Plug in your Qwen 2.5 call here (see EDIT 1)."
    )


# ======================================================================
# ==== Prompt builders =================================================
# Match these to the ACTUAL prompt template your Repair module uses,
# so arm "diag" reproduces your real Stage-1 prompt exactly.
# The ONLY difference between the arms is the presence of eps*/k*.
# ======================================================================
def build_prompt_diag(f, cwe, eps, k):
    return (
        "You are a security engineer. Fix the vulnerability in the "
        "following C/C++ function.\n"
        f"Dominant CWE: {cwe}\n"
        f"Diagnosed error source: {eps}\n"
        f"Outlier task: {k}\n"
        "Vulnerable function:\n"
        f"{f}\n"
        "Return only the corrected function."
    )


def build_prompt_cwe_only(f, cwe):
    return (
        "You are a security engineer. Fix the vulnerability in the "
        "following C/C++ function.\n"
        f"Dominant CWE: {cwe}\n"
        "Vulnerable function:\n"
        f"{f}\n"
        "Return only the corrected function."
    )


# ======================================================================
# ==== EDIT 2: load your diagnosis records =============================
# Map your stored field names to the keys used below.
# Only include TRULY-VULNERABLE flagged samples (those with a reference
# fix). Skip clean functions -- they have nothing to compare against.
# ======================================================================
def load_records(path):
    """Yield dicts with keys: id, dataset, f, cwe, eps, k, reference_fix."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # ---- adapt the right-hand sides to your field names ----
            yield {
                "id":            r["id"],
                "dataset":       r.get("dataset", "bigvul"),
                "f":             r["vulnerable_func"],   # <- your field
                "cwe":           r["dominant_cwe"],       # <- your field
                "eps":           r["error_source"],       # <- your field
                "k":             r["outlier_task"],       # <- your field
                "reference_fix": r["reference_fix"],      # <- your field
            }


# ======================================================================
def main(in_path, out_path):
    n = 0
    with open(out_path, "w") as out:
        for rec in load_records(in_path):
            f, cwe, eps, k = rec["f"], rec["cwe"], rec["eps"], rec["k"]

            # Arm A: diagnosis-guided (your method)
            patch_diag = call_qwen(build_prompt_diag(f, cwe, eps, k))
            out.write(json.dumps({
                "id": rec["id"], "arm": "diag", "dataset": rec["dataset"],
                "generated_patch": patch_diag,
                "reference_fix": rec["reference_fix"],
                "vulnerable_func": f,
            }) + "\n")

            # Arm B: CWE-only (ablation baseline)
            patch_cwe = call_qwen(build_prompt_cwe_only(f, cwe))
            out.write(json.dumps({
                "id": rec["id"], "arm": "cwe_only", "dataset": rec["dataset"],
                "generated_patch": patch_cwe,
                "reference_fix": rec["reference_fix"],
                "vulnerable_func": f,
            }) + "\n")

            n += 1
            if n % 25 == 0:
                print(f"  ...processed {n} functions", flush=True)

    print(f"Done. Wrote {out_path} for {n} functions "
          f"({2*n} patch records, 2 arms each).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python generate_ablation_arms.py "
              "diagnosis_records.jsonl results.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
