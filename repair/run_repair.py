# =============================================================================
# repair/run_repair.py  —  LLM Repair Module
#
# FRAMEWORK POSITION:
#   Preprocessing → MTD → ALC → Diagnosis → [REPAIR]
#                                                ↑
#              Triggered after Diagnosis when ALC = UNTRUSTWORTHY
#
# LLM BACKEND:
#   Local Ollama server (http://localhost:11434)
#   Model: llama3.1:70b  (or qwen2.5:32b if 70B is too slow)
#   No internet needed at runtime. Completely free.
#
# REQUIREMENT:
#   pip install ollama
#   ollama pull llama3.1:70b
#   ollama serve   (keep running in background)
#
# WHAT THIS MODULE DOES:
#   Stage 1 — Vulnerability Fix Generation
#       Reads the actual .c source file from disk (written by preprocessing).
#       Sends: function code + error_source + dominant CWE + dangerous
#       constructs + taint paths + risky lines to the LLM.
#       LLM returns a patched function with inline /* FIX [CWE-xxx] */ comments
#       on every changed line and a summary block at the top.
#
#   Stage 2 — Code Path Generation
#       LLM traces two paths using the diagnosis taint data:
#         ATTACK PATH — numbered steps showing how the bug is triggered
#         SAFE PATH   — numbered steps showing how the fix prevents it
#
#   Stage 3 — Secure Code Suggestions
#       LLM produces four ready-to-use sections:
#         SAFE API REPLACEMENTS      — e.g. strcpy → strlcpy
#         INPUT VALIDATION TEMPLATE  — reusable validation function
#         BOUNDS CHECKING PATTERN    — exact guard code before the sink
#         COMPLETE SECURE VERSION SUMMARY — numbered change list
#
# INPUTS  (from --out directory, written by previous pipeline stages):
#   diagnosis_result.json  — three-stage diagnosis (RCA + VCA + ESD)
#   mtd_result.json        — MTD outputs including source_file path
#   alc_result.json        — ALC trust decision
#
# OUTPUTS:
#   repair_result.json     — structured three-stage repair report
#   repaired_code.c        — patched function as a standalone .c file
#   repair_summary.txt     — human-readable full repair report
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPAIR_DIR = Path(__file__).resolve().parent
ROOT       = REPAIR_DIR.parent

# ── Ollama client settings ────────────────────────────────────────────────────
try:
    from ollama import Client
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME  = "llama3.1:70b"   # change to "qwen2.5:32b" if 70B is too slow
TEMPERATURE = 0.2
MAX_TOKENS  = 2048

# ── CWE mitigation catalogue — injected into every prompt ────────────────────
_CWE_MITIGATIONS = {
    "CWE-120": (
        "Replace strcpy/strcat/sprintf/gets with strlcpy/strlcat/snprintf/fgets. "
        "Always pass the destination buffer size as an explicit limit argument."
    ),
    "CWE-121": (
        "Allocate stack buffers with fixed maximum sizes. "
        "Validate all input lengths before copying to a stack buffer."
    ),
    "CWE-122": (
        "Check malloc/calloc return value before use. "
        "Verify the size argument cannot overflow before the allocation call."
    ),
    "CWE-119": (
        "Add explicit bounds checks before every buffer/array access. "
        "Use static analysis (Coverity, CodeQL) to verify all index paths."
    ),
    "CWE-125": (
        "Validate array index against array length before read. "
        "Use: assert(idx >= 0 && (size_t)idx < ARRAY_LEN) in debug builds."
    ),
    "CWE-190": (
        "Check for overflow BEFORE the operation: "
        "if (a > SIZE_MAX - b) { handle_overflow(); } "
        "Use unsigned arithmetic where possible."
    ),
    "CWE-191": (
        "Check for underflow before subtraction: if (b > a) { handle_underflow(); } "
        "Use size_t (unsigned) for sizes and lengths."
    ),
    "CWE-416": (
        "Set every pointer to NULL immediately after free(). "
        "Use: #define safe_free(p) do { free(p); (p) = NULL; } while (0)"
    ),
    "CWE-476": (
        "Check malloc/calloc/realloc return value before any use. "
        "Use a wrapper: void *xmalloc(size_t n) that aborts or returns on NULL."
    ),
    "CWE-78": (
        "Never pass user-controlled strings to system() or popen(). "
        "Use execve() with a fixed argument array and no shell interpolation."
    ),
    "CWE-134": (
        "Never use a user-controlled string as a printf format argument. "
        "Always: printf(\"%s\", user_str)  not  printf(user_str)."
    ),
    "CWE-22": (
        "Canonicalise file paths with realpath() before use. "
        "Verify the result starts with the allowed base directory."
    ),
    "CWE-252": (
        "Check return values of all I/O calls: read/write/recv/send/fclose. "
        "Treat partial reads/writes as errors."
    ),
    "CWE-330": (
        "Replace rand() with getrandom() on Linux or arc4random() on BSD/macOS. "
        "Never seed with time() for security-sensitive values."
    ),
    "CWE-704": (
        "Avoid implicit casts between signed and unsigned. "
        "Validate value is in range before explicit cast."
    ),
    "struct-array": (
        "Validate array index is within bounds before access. "
        "Guard: if (idx >= 0 && (size_t)idx < ARRAY_LEN) { ... }"
    ),
    "struct-cast": (
        "Avoid casting between unrelated struct pointer types. "
        "Use a union or tagged struct for type-safe polymorphism."
    ),
    "struct-ptr": (
        "Verify pointer arithmetic does not go past end of allocated block. "
        "Check: (ptr - base) < size"
    ),
}
_DEFAULT_MIT = (
    "Validate all inputs, check all return values, "
    "and add explicit bounds checks around pointer/array operations."
)


# =============================================================================
# Ollama LLM call
# =============================================================================

def call_llm(system_prompt: str, user_prompt: str, stage_label: str = "") -> str:
    """
    Call the local Ollama server using the ollama Python client.
    Uses the same pattern as the working client:
        client = Client(host='http://localhost:11434')
        response = client.chat(model=MODEL_NAME, messages=[...])
    """
    if not _OLLAMA_AVAILABLE:
        log.error(
            "ollama Python package not installed.\n"
            "Fix: pip install ollama"
        )
        return "[LLM ERROR: ollama package not installed — run: pip install ollama]"

    client = Client(host=OLLAMA_HOST)
    log.info(f"[{stage_label}] Calling {MODEL_NAME} on Ollama...")

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        )
        text = response["message"]["content"].strip()
        log.info(f"[{stage_label}] Response received: {len(text)} chars")
        return text

    except Exception as e:
        log.error(
            f"[{stage_label}] Ollama call failed: {e}\n"
            f"  Ensure Ollama is running:  ollama serve\n"
            f"  Ensure model is pulled:    ollama pull {MODEL_NAME}"
        )
        return f"[LLM ERROR: {e}]"


# =============================================================================
# Source code loader
# =============================================================================

def load_source(mtd_result: dict):
    """
    Load the .c source file written by preprocessing.
    mtd_result['source_file'] holds the absolute path.
    Returns (list_of_lines, path_string).
    """
    src  = mtd_result.get("source_file", "")
    name = mtd_result.get("func_name", "unknown")

    if src:
        p = Path(src)
        if p.exists():
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                log.info(f"Loaded source: {p.name}  ({len(lines)} lines)")
                return lines, str(p)
            except Exception as e:
                log.warning(f"Cannot read {p}: {e}")
        else:
            log.warning(f"Source file not found on disk: {p}")

    log.warning(f"No source available for {name} — repair will use placeholder")
    return [f"/* Source not available for function: {name} */"], src


# =============================================================================
# Stage 1 — Vulnerability Fix Generation
# =============================================================================

def stage1_fix(code_lines: list, func_name: str, diagnosis: dict) -> dict:
    """
    Ask the LLM to produce a minimal, correct patch for the function.
    Provides the full diagnosis context so the LLM targets the right lines.
    """
    log.info("=== Repair Stage 1: Vulnerability Fix Generation ===")

    rca          = diagnosis.get("stage1_root_cause_analysis",    {})
    vca          = diagnosis.get("stage2_vulnerability_context",  {})
    esd          = diagnosis.get("stage3_error_source_detection", {})
    dominant_cwe = vca.get("dominant_cwe")         or "unknown"
    error_source = esd.get("primary_error_source",  "AMBIGUOUS")
    outlier_task = rca.get("outlier_task",           "task2")
    constructs   = vca.get("dangerous_constructs",   [])
    taint        = vca.get("taint_flow",             {})
    risky_lines  = vca.get("risky_line_map",         [])
    strat_note   = vca.get("strategy_note",          "")
    mitigation   = _CWE_MITIGATIONS.get(dominant_cwe, _DEFAULT_MIT)
    func_code    = "\n".join(code_lines)

    construct_ctx = "\n".join(
        f"  Line {c.get('line','?')}: [{c.get('severity','?')}] "
        f"{c.get('type','?')} — {c.get('explanation','')}"
        for c in constructs[:5]
    ) or "  (none detected by static analysis)"

    taint_ctx = "\n".join(
        f"  {p.get('var','?')} → {p.get('sink','?')} at line {p.get('line','?')}"
        for p in taint.get("propagation_paths", [])[:4]
    ) or "  (no taint propagation paths detected)"

    risky_ctx = "\n".join(
        f"  Line {l.get('line','?')} [risk={l.get('score',0):.3f}]: "
        f"{l.get('content','')}"
        for l in risky_lines[:5]
    ) or "  (none)"

    system_prompt = (
        "You are a senior C security engineer specialising in vulnerability remediation. "
        "You will be given a C function and a full vulnerability diagnosis. "
        "Your task is to produce a MINIMAL, CORRECT patch.\n\n"
        "STRICT RULES:\n"
        "1. Fix ONLY the identified vulnerability — do not refactor unrelated code.\n"
        "2. Preserve the original function signature EXACTLY.\n"
        "3. Add an inline comment on EVERY changed line: /* FIX [CWE-xxx]: reason */\n"
        "4. Add a comment block at the very top listing all changes made.\n"
        "5. If no real vulnerability exists, return the original with a comment explaining why.\n"
        "6. Output ONLY the patched C function — no markdown, no explanation outside the code."
    )

    user_prompt = (
        f"DIAGNOSIS REPORT:\n"
        f"  Function name:   {func_name}\n"
        f"  Error source:    {error_source}\n"
        f"  Dominant CWE:    {dominant_cwe}\n"
        f"  Outlier task:    {outlier_task}\n"
        f"  Strategy note:   {strat_note}\n\n"
        f"DANGEROUS CONSTRUCTS (from static syntax analysis):\n"
        f"{construct_ctx}\n\n"
        f"TAINT PROPAGATION PATHS (from data-flow analysis):\n"
        f"{taint_ctx}\n\n"
        f"TOP RISKY LINES (from line localization):\n"
        f"{risky_ctx}\n\n"
        f"CWE MITIGATION GUIDANCE:\n"
        f"  {mitigation}\n\n"
        f"ORIGINAL FUNCTION CODE:\n"
        f"{func_code}\n\n"
        f"OUTPUT THE PATCHED FUNCTION NOW:"
    )

    patched = call_llm(system_prompt, user_prompt, "Stage1-Fix")
    patched = _strip_fences(patched)

    log.info(
        f"[Stage1] Fix generated: {len(patched)} chars  "
        f"cwe={dominant_cwe}  error_source={error_source}"
    )

    return {
        "function_name":   func_name,
        "dominant_cwe":    dominant_cwe,
        "error_source":    error_source,
        "cwe_mitigation":  mitigation,
        "patched_code":    patched,
        "patch_generated": not patched.startswith("[LLM ERROR"),
    }


# =============================================================================
# Stage 2 — Code Path Generation
# =============================================================================

def stage2_path(code_lines: list, func_name: str, diagnosis: dict) -> dict:
    """
    Ask the LLM to trace two execution paths:
      ATTACK PATH — how an attacker triggers the vulnerability
      SAFE PATH   — how the fixed code handles the same input
    """
    log.info("=== Repair Stage 2: Code Path Generation ===")

    vca          = diagnosis.get("stage2_vulnerability_context",  {})
    esd          = diagnosis.get("stage3_error_source_detection", {})
    dominant_cwe = vca.get("dominant_cwe") or "unknown"
    error_source = esd.get("primary_error_source", "AMBIGUOUS")
    taint        = vca.get("taint_flow", {})
    constructs   = vca.get("dangerous_constructs", [])
    func_code    = "\n".join(code_lines)

    sources_ctx = "\n".join(
        f"  Line {s.get('line','?')}: [{s.get('label','source')}] "
        f"{s.get('content','')}"
        for s in taint.get("top_sources", [])[:3]
    ) or "  (no taint sources identified)"

    sinks_ctx = "\n".join(
        f"  Line {s.get('line','?')}: [{s.get('label','sink')}] "
        f"{s.get('content','')}"
        for s in taint.get("top_sinks", [])[:3]
    ) or "  (no taint sinks identified)"

    paths_ctx = "\n".join(
        f"  {p.get('var','?')} → {p.get('sink','?')} "
        f"at line {p.get('line','?')}"
        for p in taint.get("propagation_paths", [])[:3]
    ) or "  (no propagation paths)"

    constructs_ctx = "\n".join(
        f"  Line {c.get('line','?')}: {c.get('type','?')} "
        f"[{c.get('severity','?')}] — {c.get('explanation','')}"
        for c in constructs[:3]
    ) or "  (none)"

    system_prompt = (
        "You are a security researcher tracing vulnerability execution paths in C code. "
        "Given a C function and its vulnerability context, produce exactly "
        "TWO clearly labelled sections.\n\n"
        "SECTION 1 — ATTACK PATH:\n"
        "  Numbered steps showing how an attacker triggers the vulnerability. "
        "  Reference specific line numbers and variable names from the code.\n\n"
        "SECTION 2 — SAFE PATH:\n"
        "  Numbered steps showing how the FIXED code handles the same attacker input. "
        "  Reference the specific check or validation that blocks the attack.\n\n"
        "Be concrete and technical. Use exact variable names and line numbers. "
        "No markdown. Plain text only."
    )

    user_prompt = (
        f"FUNCTION: {func_name}\n"
        f"DOMINANT CWE: {dominant_cwe}\n"
        f"ERROR SOURCE: {error_source}\n\n"
        f"TAINT SOURCES (where untrusted data enters):\n"
        f"{sources_ctx}\n\n"
        f"TAINT SINKS (where data reaches dangerous operations):\n"
        f"{sinks_ctx}\n\n"
        f"PROPAGATION PATHS:\n"
        f"{paths_ctx}\n\n"
        f"UNSAFE CONSTRUCTS:\n"
        f"{constructs_ctx}\n\n"
        f"FUNCTION CODE:\n"
        f"{func_code}\n\n"
        f"ATTACK PATH:\n\n"
        f"SAFE PATH:"
    )

    response    = call_llm(system_prompt, user_prompt, "Stage2-Path")
    attack_path = _extract_section(response, "ATTACK PATH")
    safe_path   = _extract_section(response, "SAFE PATH")

    log.info(
        f"[Stage2] attack={len(attack_path)} chars  safe={len(safe_path)} chars"
    )

    return {
        "function_name":      func_name,
        "dominant_cwe":       dominant_cwe,
        "taint_sources":      taint.get("top_sources",       []),
        "taint_sinks":        taint.get("top_sinks",         []),
        "propagation_paths":  taint.get("propagation_paths", []),
        "attack_path":        attack_path or response,
        "safe_path":          safe_path,
        "full_path_analysis": response,
    }


# =============================================================================
# Stage 3 — Secure Code Suggestions
# =============================================================================

def stage3_suggest(code_lines: list, func_name: str, diagnosis: dict) -> dict:
    """
    Ask the LLM to produce four practical, copy-paste-ready secure coding
    suggestions tailored to this function and its detected CWEs.
    """
    log.info("=== Repair Stage 3: Secure Code Suggestions ===")

    vca          = diagnosis.get("stage2_vulnerability_context",  {})
    esd          = diagnosis.get("stage3_error_source_detection", {})
    dominant_cwe = vca.get("dominant_cwe")         or "unknown"
    error_source = esd.get("primary_error_source",  "AMBIGUOUS")
    cwe_context  = vca.get("cwe_context",            {})
    complexity   = vca.get("complexity_note",         "")
    strategy     = vca.get("strategy",                "heuristic")
    repair_acts  = esd.get("recommended_repair_actions", [])
    func_code    = "\n".join(code_lines)

    cwe_block = "\n".join(
        f"  {cwe} [{info.get('severity','?')}]: {info.get('description','')}\n"
        f"    Fix: {_CWE_MITIGATIONS.get(cwe, _DEFAULT_MIT)}"
        for cwe, info in cwe_context.items()
    ) or f"  {dominant_cwe}: {_CWE_MITIGATIONS.get(dominant_cwe, _DEFAULT_MIT)}"

    actions_block = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(repair_acts)
    ) or "  (no specific actions recommended)"

    system_prompt = (
        "You are a C/C++ secure coding expert. "
        "Produce PRACTICAL, COPY-PASTE-READY secure code suggestions.\n\n"
        "Write EXACTLY FOUR sections with these exact headers:\n\n"
        "SAFE API REPLACEMENTS:\n"
        "  Show UNSAFE call on left, SAFE replacement on right. "
        "  Include required #include headers.\n\n"
        "INPUT VALIDATION TEMPLATE:\n"
        "  A reusable C function or macro validating this function's inputs. "
        "  Under 20 lines. C99.\n\n"
        "BOUNDS CHECKING PATTERN:\n"
        "  Exact guard code to add before the vulnerable operation. Under 10 lines.\n\n"
        "COMPLETE SECURE VERSION SUMMARY:\n"
        "  Numbered list of ALL changes needed. One line each.\n\n"
        "No markdown. No code fences. C99 only."
    )

    user_prompt = (
        f"FUNCTION: {func_name}\n"
        f"ERROR SOURCE: {error_source}\n"
        f"STRATEGY: {strategy}\n"
        f"COMPLEXITY: {complexity}\n\n"
        f"CWE VULNERABILITIES AND MITIGATIONS:\n"
        f"{cwe_block}\n\n"
        f"DIAGNOSIS-RECOMMENDED ACTIONS:\n"
        f"{actions_block}\n\n"
        f"ORIGINAL FUNCTION CODE:\n"
        f"{func_code}\n\n"
        f"SAFE API REPLACEMENTS:\n\n"
        f"INPUT VALIDATION TEMPLATE:\n\n"
        f"BOUNDS CHECKING PATTERN:\n\n"
        f"COMPLETE SECURE VERSION SUMMARY:"
    )

    response         = call_llm(system_prompt, user_prompt, "Stage3-Suggest")
    api_replacements = _extract_section(response, "SAFE API REPLACEMENTS")
    input_validation = _extract_section(response, "INPUT VALIDATION TEMPLATE")
    bounds_checking  = _extract_section(response, "BOUNDS CHECKING PATTERN")
    complete_secure  = _extract_section(response, "COMPLETE SECURE VERSION SUMMARY")

    log.info(f"[Stage3] Suggestions: {len(response)} chars total")

    return {
        "function_name":    func_name,
        "error_source":     error_source,
        "dominant_cwe":     dominant_cwe,
        "cwe_context":      cwe_context,
        "api_replacements": api_replacements,
        "input_validation": input_validation,
        "bounds_checking":  bounds_checking,
        "complete_secure":  complete_secure,
        "full_suggestions": response,
    }


# =============================================================================
# Helpers
# =============================================================================

def _strip_fences(text: str) -> str:
    """Remove markdown code fences the LLM might accidentally include."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("```")
    ).strip()


def _extract_section(text: str, header: str) -> str:
    """
    Extract a named section from LLM output.
    Finds the header line and returns text until the next known section header.
    """
    if not text:
        return ""

    all_headers = [
        "attack path", "safe path",
        "safe api replacements", "input validation template",
        "bounds checking pattern", "complete secure version summary",
    ]
    lines        = text.splitlines()
    inside       = False
    result       = []
    header_lower = header.lower()

    for line in lines:
        stripped_lower = line.strip().lower().rstrip(":")
        if header_lower in stripped_lower:
            inside = True
            continue
        if inside:
            is_next_header = any(
                h in stripped_lower
                for h in all_headers
                if h != header_lower and stripped_lower
            )
            if is_next_header:
                break
            result.append(line)

    return "\n".join(result).strip()


# =============================================================================
# Summary writer
# =============================================================================

def write_summary(repair: dict, out_dir: Path):
    s    = repair
    fix  = s["stage1_vulnerability_fix"]
    path = s["stage2_code_path"]
    sug  = s["stage3_secure_suggestions"]

    lines = [
        "=" * 70,
        f"  LLM REPAIR REPORT  (Ollama / {s['llm_model']})",
        f"  Sample:  {s['sample_id']}  |  Dataset: {s['dataset']}",
        f"  MTD:     {s['mtd_verdict']}  V={s['vulnerability_score']:.4f}",
        f"  ALC:     {s['alc_decision'].upper()}  T={s['trust_score']:.4f}",
        "=" * 70, "",
        "── STAGE 1: VULNERABILITY FIX ────────────────────────────────────────",
        f"  Function:     {fix['function_name']}",
        f"  Dominant CWE: {fix['dominant_cwe']}",
        f"  Error source: {fix['error_source']}",
        f"  Mitigation:   {fix['cwe_mitigation']}",
        f"  Patch OK:     {fix['patch_generated']}",
        "",
        "  PATCHED CODE:",
        "-" * 60,
        fix["patched_code"],
        "-" * 60, "",
        "── STAGE 2: CODE PATH ────────────────────────────────────────────────",
        f"  Function:     {path['function_name']}",
        f"  Dominant CWE: {path['dominant_cwe']}",
        "",
        "  ATTACK PATH:",
        path["attack_path"],
        "",
        "  SAFE PATH (after fix):",
        path["safe_path"], "",
        "── STAGE 3: SECURE CODE SUGGESTIONS ─────────────────────────────────",
        f"  Dominant CWE: {sug['dominant_cwe']}", "",
        "  SAFE API REPLACEMENTS:",
        sug["api_replacements"], "",
        "  INPUT VALIDATION TEMPLATE:",
        sug["input_validation"], "",
        "  BOUNDS CHECKING PATTERN:",
        sug["bounds_checking"], "",
        "  COMPLETE SECURE VERSION SUMMARY:",
        sug["complete_secure"], "",
        "=" * 70,
    ]

    p = out_dir / "repair_summary.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Repair summary → {p}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Declare global first — before any reference to MODEL_NAME in this scope
    global MODEL_NAME

    parser = argparse.ArgumentParser(
        description=(
            f"LLM Repair Module — Ollama local backend\n"
            f"Model:    {MODEL_NAME}\n"
            f"Requires: pip install ollama\n"
            f"          ollama pull {MODEL_NAME}\n"
            f"          ollama serve"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory (must contain diagnosis_result.json, "
             "mtd_result.json, alc_result.json)"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Ollama model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run repair even if ALC=TRUSTWORTHY"
    )
    args    = parser.parse_args()
    out_dir = Path(args.out)

    # Apply model override from CLI
    MODEL_NAME = args.model

    # ── Check ollama package ──────────────────────────────────────────────────
    if not _OLLAMA_AVAILABLE:
        log.error(
            "ollama Python package not installed.\n"
            "Fix: pip install ollama"
        )
        sys.exit(1)

    # ── Check Ollama server is reachable and model is available ───────────────
    try:
        client = Client(host=OLLAMA_HOST)
        models = client.list()

        # The ollama Python library changed its response format across versions:
        #   old (dict):   models["models"]  = [{"name": "llama3.1:70b", ...}]
        #   new (object): models.models     = [Model(model="llama3.1:70b", ...)]
        raw_list = []
        if isinstance(models, dict):
            raw_list = models.get("models", [])
        elif hasattr(models, "models"):
            raw_list = models.models or []

        available = []
        for m in raw_list:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model", "")
            elif hasattr(m, "model"):
                name = m.model
            elif hasattr(m, "name"):
                name = m.name
            else:
                name = str(m)
            if name:
                available.append(name)

        log.info(f"Ollama running. Available models: {available}")

        if available and not any(MODEL_NAME in m for m in available):
            log.warning(
                f"Model '{MODEL_NAME}' not listed in Ollama — attempting anyway.\n"
                f"  Pull it:   ollama pull {MODEL_NAME}\n"
                f"  Available: {available}"
            )

    except Exception as e:
        log.error(
            f"Cannot reach Ollama at {OLLAMA_HOST}: {e}\n"
            f"  Start it:  ollama serve\n"
            f"  Install:   curl -fsSL https://ollama.ai/install.sh | sh"
        )
        sys.exit(1)

    # ── Check required input files ────────────────────────────────────────────
    for fname in ("diagnosis_result.json", "mtd_result.json", "alc_result.json"):
        if not (out_dir / fname).exists():
            log.error(f"{fname} not found in {out_dir}")
            sys.exit(1)

    diagnosis  = json.loads(
        (out_dir / "diagnosis_result.json").read_text(encoding="utf-8"))
    mtd_result = json.loads(
        (out_dir / "mtd_result.json").read_text(encoding="utf-8"))
    alc_result = json.loads(
        (out_dir / "alc_result.json").read_text(encoding="utf-8"))

    alc_decision = alc_result.get("decision",    "untrustworthy")
    mtd_verdict  = alc_result.get("mtd_verdict", "UNKNOWN")

    if diagnosis.get("skipped") and not args.force:
        log.info("Diagnosis skipped (ALC=TRUSTWORTHY). Repair also skipped.")
        (out_dir / "repair_result.json").write_text(
            json.dumps({"skipped": True, "reason": "ALC=TRUSTWORTHY"}, indent=2),
            encoding="utf-8",
        )
        sys.exit(0)

    sample_id = diagnosis.get("sample_id", "?")
    dataset   = diagnosis.get("dataset",   "unknown")
    V         = float(diagnosis.get("vulnerability_score", 0.0))
    T         = float(diagnosis.get("trust_score",         0.0))
    func_name = mtd_result.get("func_name",  "unknown")
    strategy  = mtd_result.get("strategy",   "heuristic")

    log.info(
        f"Repair | id={sample_id}  dataset={dataset}  func={func_name}  "
        f"MTD={mtd_verdict}  V={V:.4f}  ALC={alc_decision.upper()}  T={T:.4f}  "
        f"model={MODEL_NAME}"
    )

    code_lines, _ = load_source(mtd_result)

    # ── Run three repair stages ───────────────────────────────────────────────
    fix_result  = stage1_fix(code_lines,     func_name, diagnosis)
    path_result = stage2_path(code_lines,    func_name, diagnosis)
    sug_result  = stage3_suggest(code_lines, func_name, diagnosis)

    log.info(
        f"All stages complete — "
        f"cwe={fix_result['dominant_cwe']}  "
        f"error_source={fix_result['error_source']}  "
        f"patch_ok={fix_result['patch_generated']}"
    )

    repair_result = {
        "sample_id":            sample_id,
        "dataset":              dataset,
        "strategy":             strategy,
        "mtd_verdict":          mtd_verdict,
        "vulnerability_score":  V,
        "trust_score":          T,
        "alc_decision":         alc_decision,
        "llm_model":            MODEL_NAME,
        "llm_backend":          "ollama_local",

        "stage1_vulnerability_fix":  fix_result,
        "stage2_code_path":          path_result,
        "stage3_secure_suggestions": sug_result,

        "repair_verdict": {
            "function_patched":    func_name,
            "dominant_cwe":        fix_result["dominant_cwe"],
            "error_source":        fix_result["error_source"],
            "patch_generated":     fix_result["patch_generated"],
            "patched_code_length": len(fix_result["patched_code"]),
        },
    }

    (out_dir / "repair_result.json").write_text(
        json.dumps(repair_result, indent=2), encoding="utf-8"
    )

    if fix_result["patch_generated"]:
        c_path = out_dir / "repaired_code.c"
        c_path.write_text(
            f"/* ============================================================\n"
            f"   Repaired by LLM Repair Module (Ollama / {MODEL_NAME})\n"
            f"   Sample:       {sample_id}  |  Dataset: {dataset}\n"
            f"   Function:     {func_name}\n"
            f"   Dominant CWE: {fix_result['dominant_cwe']}\n"
            f"   Error source: {fix_result['error_source']}\n"
            f"   MTD verdict:  {mtd_verdict}  V={V:.4f}\n"
            f"   ALC decision: {alc_decision.upper()}  T={T:.4f}\n"
            f"   ============================================================ */\n\n"
            + fix_result["patched_code"],
            encoding="utf-8",
        )
        log.info(f"Patched code → {c_path}")

    write_summary(repair_result, out_dir)
    log.info(f"All repair outputs written → {out_dir}")


if __name__ == "__main__":
    main()


    