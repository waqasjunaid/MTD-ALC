# =============================================================================
# preprocessing/suspicious_line_mapper.py
#
# Suspicious Line Mapping
#
# Takes a function dict (from function_extractor.py) and flaw line data
# (from the dataset) and produces a rich line-level map used by the MTD.
#
# Three mapping strategies are supported:
#
#   1. GROUND_TRUTH  — dataset provides exact flaw_line_index values (BigVul)
#   2. HEURISTIC     — no ground truth; flag lines that match risk patterns
#   3. ALL_LINES     — mark every line (MegaVul fallback; lowest signal quality)
#
# Output schema (LineMap):
#   {
#       "func_id":        str,
#       "source_file":    str,
#       "strategy":       "ground_truth" | "heuristic" | "all_lines",
#       "total_lines":    int,
#       "suspicious_lines": [
#           {
#               "line_no":   int,       # 1-based absolute line in the function
#               "content":   str,       # stripped source text of that line
#               "reason":    str,       # why it was flagged
#               "confidence": float,   # 0.0 – 1.0
#           },
#           ...
#       ],
#       "suspicious_line_numbers": [int, ...],   # flat list for MTD input
#   }
# =============================================================================

import re
import logging
from typing import Literal, Optional, List
from pathlib import Path

log = logging.getLogger(__name__)

Strategy = Literal["ground_truth", "heuristic", "all_lines"]

# ---------------------------------------------------------------------------
# Heuristic risk rules
# Each rule: (compiled pattern, reason label, confidence score)
# ---------------------------------------------------------------------------
_HEURISTIC_RULES: list[tuple[re.Pattern, str, float]] = [

    # --- Memory safety ---
    (re.compile(r"\bstrcpy\s*\("),          "unsafe strcpy",            0.90),
    (re.compile(r"\bstrcat\s*\("),          "unsafe strcat",            0.85),
    (re.compile(r"\bgets\s*\("),            "unsafe gets",              0.95),
    (re.compile(r"\bsprintf\s*\("),         "unsafe sprintf",           0.80),
    (re.compile(r"\bscanf\s*\("),           "unsafe scanf",             0.75),
    (re.compile(r"\bmemcpy\s*\("),          "unchecked memcpy",         0.65),
    (re.compile(r"\bmemmove\s*\("),         "unchecked memmove",        0.60),
    (re.compile(r"\balloca\s*\("),          "stack alloca",             0.80),

    # --- Integer issues ---
    (re.compile(r"\batoi\s*\("),            "atoi no error check",      0.70),
    (re.compile(r"\batol\s*\("),            "atol no error check",      0.70),
    (re.compile(r"\(int\)\s*strlen"),       "signed/unsigned cast",     0.75),
    (re.compile(r"<<\s*\d+"),               "bit shift — check bounds", 0.55),

    # --- NULL / pointer risks ---
    (re.compile(r"\bmalloc\s*\("),          "malloc without NULL check",0.65),
    (re.compile(r"\brealloc\s*\("),         "realloc without NULL check",0.70),
    (re.compile(r"\bfree\s*\("),            "free — check double-free", 0.55),
    (re.compile(r"\bNULL\b.*==|==.*\bNULL\b"), "NULL comparison",       0.40),

    # --- Format string ---
    (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'),  "printf with var format",0.85),
    (re.compile(r'\bfprintf\s*\(\s*\w+\s*,\s*\w+\s*\)'), "fprintf var fmt", 0.80),
    (re.compile(r'\bsyslog\s*\(\s*\w+\s*,\s*\w+\s*\)'),  "syslog var fmt",  0.80),

    # --- Use after free / dangling ---
    (re.compile(r"\bfree\s*\(\w+\)"),          "free — check use-after-free",  0.65),

    # --- Command injection ---
    (re.compile(r"\bsystem\s*\("),          "system() call",            0.80),
    (re.compile(r"\bpopen\s*\("),           "popen() call",             0.80),
    (re.compile(r"\bexecve?\s*\("),         "execv/execve call",        0.75),

    # --- File / path ---
    (re.compile(r"\bopen\s*\("),            "open() — check path",      0.50),
    (re.compile(r"\bfopen\s*\("),           "fopen() — check path",     0.50),
    (re.compile(r'"\.\./'),                 "path traversal pattern",   0.85),

    # --- Crypto / randomness ---
    (re.compile(r"\brand\s*\("),            "weak rand()",              0.70),
    (re.compile(r"\bsrand\s*\(\s*time"),    "predictable seed",         0.75),

    # --- Return value ignored ---
    (re.compile(r"^\s*read\s*\("),          "read() return ignored",    0.65),
    (re.compile(r"^\s*write\s*\("),         "write() return ignored",   0.60),
    (re.compile(r"^\s*recv\s*\("),          "recv() return ignored",    0.65),
    (re.compile(r"^\s*send\s*\("),          "send() return ignored",    0.60),

    # --- Array indexing ---
    (re.compile(r"\w+\s*\[\s*\w+\s*\]"),   "array index — check bounds",0.45),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_suspicious_lines(
    func: dict,
    flaw_lines: Optional[List[int]] = None,
    strategy: Optional[Strategy] = None,
) -> dict:
    """
    Build a suspicious line map for one function.

    Args:
        func:       Function dict from function_extractor.py.
        flaw_lines: Ground-truth 1-based line numbers relative to the
                    original file (converted to function-relative internally).
                    Pass None or [] to trigger heuristic / all_lines fallback.
        strategy:   Override automatic strategy selection.
                    If None, strategy is chosen automatically:
                      - flaw_lines provided and non-empty → ground_truth
                      - otherwise                         → heuristic

    Returns:
        LineMap dict (see module docstring).
    """
    code_lines = func["code"].splitlines()
    total      = len(code_lines)
    func_start = func.get("start_line", 1)   # 1-based start in original file

    # --- Determine strategy ---
    if strategy is None:
        if flaw_lines:
            strategy = "ground_truth"
        else:
            strategy = "heuristic"

    log.info(
        f"[{func['id']}] Mapping suspicious lines — strategy={strategy}, "
        f"total_lines={total}"
    )

    if strategy == "ground_truth":
        entries = _map_ground_truth(code_lines, flaw_lines, func_start)

    elif strategy == "heuristic":
        entries = _map_heuristic(code_lines)

    else:  # all_lines
        entries = _map_all_lines(code_lines)

    # Sort by line number
    entries.sort(key=lambda e: e["line_no"])

    line_map = {
        "func_id":                 func["id"],
        "source_file":             func.get("source_file"),
        "strategy":                strategy,
        "total_lines":             total,
        "suspicious_lines":        entries,
        "suspicious_line_numbers": [e["line_no"] for e in entries],
    }

    log.info(
        f"[{func['id']}] {len(entries)}/{total} lines flagged "
        f"({strategy})"
    )
    return line_map


def map_batch(
    funcs: list[dict],
    flaw_lines_list: Optional[List[List[int]]] = None,
    strategy: Optional[Strategy] = None,
) -> list[dict]:
    """
    Map suspicious lines for a batch of functions.

    Args:
        funcs:            List of function dicts.
        flaw_lines_list:  Parallel list of flaw_lines per function.
                          Pass None to use heuristic for all.
        strategy:         Override for all functions.

    Returns:
        List of LineMap dicts.
    """
    results = []
    for i, func in enumerate(funcs):
        fl = (flaw_lines_list[i] if flaw_lines_list else None)
        try:
            results.append(map_suspicious_lines(func, fl, strategy))
        except Exception as e:
            log.error(f"[{func.get('id', i)}] Mapping failed: {e}")
            results.append(None)
    return results


def summarise(line_map: dict) -> str:
    """Return a human-readable one-line summary of a LineMap."""
    n   = len(line_map["suspicious_line_numbers"])
    tot = line_map["total_lines"]
    pct = (n / tot * 100) if tot else 0
    return (
        f"{line_map['func_id']}: {n}/{tot} lines suspicious "
        f"({pct:.1f}%) [{line_map['strategy']}]"
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _map_ground_truth(
    code_lines: list[str],
    flaw_lines: list[int],
    func_start: int,
) -> list[dict]:
    """
    Map ground-truth flaw line numbers to entries in the function body.

    For dataset rows (BigVul), flaw_lines are already function-relative
    1-based numbers produced by prepare_bigvul.py, and func_start is always
    1 (set by extract_from_dataset_row). No offset conversion is needed or
    applied — flaw_lines are used directly as function-relative line numbers.

    For multi-function source files where func_start > 1, the offset
    conversion (abs_line - func_start + 1) is still applied correctly.
    """
    entries = []
    flaw_set = set(flaw_lines)

    for abs_line in sorted(flaw_set):
        # When func_start == 1 (all dataset rows), rel_line == abs_line.
        # When func_start > 1 (multi-function files), converts to function-relative.
        rel_line = abs_line - func_start + 1

        if rel_line < 1 or rel_line > len(code_lines):
            log.warning(
                f"Flaw line {abs_line} (function-relative {rel_line}) is outside "
                f"function body of {len(code_lines)} lines — skipping"
            )
            continue

        content = code_lines[rel_line - 1].strip()
        entries.append({
            "line_no":    rel_line,
            "content":    content,
            "reason":     "ground_truth_flaw_line",
            "confidence": 1.0,
        })

    return entries


def _map_heuristic(code_lines: list[str]) -> list[dict]:
    """
    Apply pattern-based rules to each line and collect matches.
    A line can match multiple rules; we keep the highest confidence reason.
    """
    # line_no (1-based) → best match so far
    best: dict[int, dict] = {}

    for line_no, raw_line in enumerate(code_lines, start=1):
        stripped = raw_line.strip()

        # Skip blank lines and pure comments
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue

        for pattern, reason, confidence in _HEURISTIC_RULES:
            if pattern.search(raw_line):
                existing = best.get(line_no)
                if existing is None or confidence > existing["confidence"]:
                    best[line_no] = {
                        "line_no":    line_no,
                        "content":    stripped,
                        "reason":     reason,
                        "confidence": confidence,
                    }

    return list(best.values())


def _map_all_lines(code_lines: list[str]) -> list[dict]:
    """
    Mark every non-blank, non-comment line as suspicious.
    Used as a last-resort fallback when no ground truth or heuristics apply.
    Confidence is uniformly low (0.10) to signal low information quality.
    """
    entries = []
    for line_no, raw_line in enumerate(code_lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue
        entries.append({
            "line_no":    line_no,
            "content":    stripped,
            "reason":     "all_lines_fallback",
            "confidence": 0.10,
        })
    return entries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python suspicious_line_mapper.py <func_json_file> [flaw_lines...]")
        print("  flaw_lines: space-separated 1-based line numbers (optional)")
        sys.exit(1)

    func_data  = json.loads(Path(sys.argv[1]).read_text())
    flaw_input = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else None

    result = map_suspicious_lines(func_data, flaw_input)
    print(json.dumps(result, indent=2))
    print(f"\n{summarise(result)}")