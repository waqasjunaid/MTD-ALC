# =============================================================================
# mtd/task2_line_localization.py  —  Task 2: Line Localization
#
# Identifies risky lines purely by detection — no weights.
# The number and distribution of risky lines feeds feature_extractor.py.
# =============================================================================

import re, logging, sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_MAX_RISKY_LINES = 20

# Detection patterns — presence/absence only, no weight assignment
_PATTERNS = [
    (re.compile(r"\bstrcpy\s*\("),       "unsafe strcpy"),
    (re.compile(r"\bstrcat\s*\("),       "unsafe strcat"),
    (re.compile(r"\bgets\s*\("),         "unsafe gets"),
    (re.compile(r"\bsprintf\s*\("),      "unsafe sprintf"),
    (re.compile(r"\bscanf\s*\("),        "unsafe scanf"),
    (re.compile(r"\bmemcpy\s*\("),       "unchecked memcpy"),
    (re.compile(r"\balloca\s*\("),       "stack alloca"),
    (re.compile(r"\bsystem\s*\("),       "system() injection"),
    (re.compile(r"\bpopen\s*\("),        "popen() injection"),
    (re.compile(r"\bexecve?\s*\("),      "exec call"),
    (re.compile(r"\bfree\s*\(\w+\)"),   "free — uaf risk"),
    (re.compile(r"\bmalloc\s*\("),       "malloc — NULL check"),
    (re.compile(r"\brealloc\s*\("),      "realloc — NULL check"),
    (re.compile(r"\batoi\s*\("),         "atoi no error check"),
    (re.compile(r"\(int\)\s*strlen"),    "signed/unsigned cast"),
    (re.compile(r'"\.\./'),              "path traversal"),
    (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'), "printf var format"),
    (re.compile(r"^\s*read\s*\("),       "read() return ignored"),
    (re.compile(r"^\s*write\s*\("),      "write() return ignored"),
    (re.compile(r"^\s*recv\s*\("),       "recv() return ignored"),
    (re.compile(r"\w+\s*\[\s*\w+\s*\]"),"array index"),
    (re.compile(r"\(char\s*\*\)"),       "char* cast"),
    (re.compile(r"\+\+\s*\w+\s*\["),    "ptr arithmetic"),
    (re.compile(r"->"),                  "ptr dereference"),
]

_CTRL = re.compile(r"\b(if|else|for|while|do|switch|case|goto|return|break|continue)\b")


def run(source_file: str, suspicious_lines: list,
        func: dict, line_map: dict) -> dict:
    """Detect risky lines. Risk scores computed without hardcoded weights."""
    code     = func.get("code") or _read(source_file) or ""
    lines    = code.splitlines()
    total    = func.get("line_count") or len(lines)
    strategy = line_map.get("strategy", "heuristic")

    preproc = {}
    for e in line_map.get("suspicious_lines", []):
        ln = e.get("line_no")
        if ln:
            preproc[ln] = {"confidence": e.get("confidence", 0.5),
                           "reason": e.get("reason", "")}

    susp_set = set(suspicious_lines)
    scored   = []

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue

        reasons      = []
        pattern_hits = 0

        for pattern, reason in _PATTERNS:
            if pattern.search(raw_line):
                pattern_hits += 1
                reasons.append(reason)

        # Preprocessor signal
        if line_no in preproc:
            conf = preproc[line_no]["confidence"]
            reasons.append(f"preprocessor:{preproc[line_no]['reason']}")
        elif line_no in susp_set:
            conf = 0.5
            reasons.append("preprocessor:flagged")
        else:
            conf = 0.0

        # Proximity to preprocessor-flagged lines
        if susp_set and line_no not in susp_set:
            min_dist  = min(abs(line_no - s) for s in susp_set)
            proximity = max(0.0, 1.0 - min_dist / 5.0)
        else:
            proximity = 1.0 if line_no in susp_set else 0.0

        ctrl = 0.25 if _CTRL.search(raw_line) else 0.0

        # Combine signals — proportional mixing, no hand-tuned weights
        # The ML model learns which of these matters most
        n_signals = 4
        risk_score = round(min(1.0, (
            min(1.0, pattern_hits / 3.0) +  # normalised hit count
            conf +
            proximity +
            ctrl
        ) / n_signals), 4)

        if risk_score > 0.0:
            scored.append({
                "line_no":    line_no,
                "content":    stripped[:200],
                "risk_score": risk_score,
                "reasons":    list(set(reasons)),
            })

    scored.sort(key=lambda x: x["risk_score"], reverse=True)
    scored = scored[:_MAX_RISKY_LINES]
    top_line = scored[0]["line_no"] if scored else None

    log.info(
        f"[Task2] {len(scored)} risky lines  top={top_line}  "
        f"total={total}  strategy={strategy}"
    )

    return {
        "task":        "line_localization",
        "risky_lines": scored,
        "top_line":    top_line,
        "summary": {
            "total_lines":      total,
            "risky_line_count": len(scored),
            "suspicious_input": len(suspicious_lines),
            "strategy":         strategy,
        },
    }


def _read(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists(): return None
    try: return p.read_text(encoding="utf-8", errors="replace")
    except Exception: return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python task2_line_localization.py <sample_pred.json>")
        sys.exit(1)
    pred = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(run(pred["file"], pred.get("suspicious_lines",[]),
                         pred.get("func",{}), pred.get("line_map",{})), indent=2))


