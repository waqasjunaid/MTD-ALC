# # =============================================================================
# # mtd/task3_syntax_risk_prediction.py  —  Task 3: Syntax Risk Prediction
# #
# # Detects unsafe constructs and classifies them by severity tier.
# # No base severity scores — the ML model learns severity importance.
# # Output: counts per severity tier + CWE tags → fed to feature_extractor.
# # =============================================================================
#
# import re, logging, math, sys
# from collections import Counter
# from pathlib import Path
# from typing import Optional
#
# log = logging.getLogger(__name__)
#
# # Detection catalogue — (pattern, construct_type, severity_tier, cwe, explanation)
# # No numeric scores — severity is categorical only (HIGH/MEDIUM/LOW)
# _CONSTRUCTS = [
#     # HIGH
#     (re.compile(r"\bgets\s*\("),                  "unbounded_copy",       "HIGH", "CWE-120", "gets() — no bounds"),
#     (re.compile(r"\bstrcpy\s*\("),                "unbounded_copy",       "HIGH", "CWE-120", "strcpy() — no bounds"),
#     (re.compile(r"\bstrcat\s*\("),                "unbounded_copy",       "HIGH", "CWE-120", "strcat() — no bounds"),
#     (re.compile(r"\bsystem\s*\(\s*(?!\s*\")"),    "command_injection",    "HIGH", "CWE-78",  "system() variable arg"),
#     (re.compile(r"\bpopen\s*\(\s*(?!\s*\")"),     "command_injection",    "HIGH", "CWE-78",  "popen() variable arg"),
#     (re.compile(r"\bexecve?\s*\(\s*(?!\s*\")"),   "command_injection",    "HIGH", "CWE-78",  "exec() variable arg"),
#     (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'),     "format_string_inject", "HIGH", "CWE-134", "printf() variable format"),
#     (re.compile(r'\bfprintf\s*\(\s*\w+\s*,\s*\w+\s*\)'), "format_string_inject","HIGH","CWE-134","fprintf() variable format"),
#     (re.compile(r'\bsyslog\s*\(\s*\w+\s*,\s*\w+\s*\)'),  "format_string_inject","HIGH","CWE-134","syslog() variable format"),
#     (re.compile(r'"\.\./'),                       "path_traversal",       "HIGH", "CWE-22",  "hardcoded ../ path traversal"),
#     # MEDIUM
#     (re.compile(r"\bsprintf\s*\("),               "unbounded_copy",       "MEDIUM","CWE-120","sprintf() — use snprintf"),
#     (re.compile(r"\bscanf\s*\("),                 "unbounded_copy",       "MEDIUM","CWE-120","scanf() — add width specifier"),
#     (re.compile(r"\bmalloc\s*\([^)]+\)\s*;"),     "unchecked_alloc",      "MEDIUM","CWE-476","malloc() result unchecked"),
#     (re.compile(r"\brealloc\s*\("),               "unchecked_alloc",      "MEDIUM","CWE-476","realloc() — original ptr lost"),
#     (re.compile(r"\(int\)\s*strlen"),             "integer_truncation",   "MEDIUM","CWE-190","Casting size_t to int"),
#     (re.compile(r"\(short\)\s*\w+"),              "integer_truncation",   "MEDIUM","CWE-190","Narrowing cast to short"),
#     (re.compile(r"\(char\s*\*\)\s*(?!\"|\()"),    "dangerous_cast",       "MEDIUM","CWE-704","Cast to char* — type safety lost"),
#     (re.compile(r"\(void\s*\*\)\s*\w+"),          "dangerous_cast",       "MEDIUM","CWE-704","Cast to void* — loses type info"),
#     (re.compile(r"\batoi\s*\("),                  "missing_return_check", "MEDIUM","CWE-190","atoi() error indistinguishable"),
#     (re.compile(r"\balloca\s*\("),                "unbounded_copy",       "MEDIUM","CWE-121","alloca() — no size bound"),
#     (re.compile(r"^\s*read\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","read() return ignored"),
#     (re.compile(r"^\s*write\s*\(", re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","write() return ignored"),
#     (re.compile(r"^\s*recv\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","recv() return ignored"),
#     (re.compile(r"^\s*send\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","send() return ignored"),
#     # LOW
#     (re.compile(r"\bsnprintf\s*\("),              "unbounded_copy",       "LOW",  "CWE-120","snprintf() — check truncation"),
#     (re.compile(r"\bstrncat\s*\("),               "unbounded_copy",       "LOW",  "CWE-120","strncat() — check total length"),
#     (re.compile(r"<<\s*\d+"),                     "integer_truncation",   "LOW",  "CWE-190","Bit shift — verify width"),
#     (re.compile(r"\w+\+\+\s*\[|\[\s*\w+\+\+\s*\]"),"pointer_arithmetic", "LOW",  "CWE-119","Increment inside array index"),
#     (re.compile(r"\brand\s*\("),                  "weak_randomness",      "LOW",  "CWE-330","rand() — not cryptographic"),
#     (re.compile(r"\bsrand\s*\(\s*time"),          "weak_randomness",      "LOW",  "CWE-330","srand(time()) — predictable"),
# ]
#
# _CASE_PAT  = re.compile(r"^\s*case\s+.+:")
# _BREAK_PAT = re.compile(r"\b(break|return|goto|continue)\b")
#
#
# def run(source_file: str, suspicious_lines: list,
#         func: dict, line_map: dict) -> dict:
#     """Detect unsafe constructs and count by severity. No scores — counts only."""
#     code     = func.get("code") or _read(source_file) or ""
#     lines    = code.splitlines()
#     strategy = line_map.get("strategy", "heuristic")
#
#     preproc = {e["line_no"]: e.get("confidence", 0.5)
#                for e in line_map.get("suspicious_lines", []) if e.get("line_no")}
#     susp_set = set(suspicious_lines)
#
#     detected = []
#     for line_no, raw_line in enumerate(lines, start=1):
#         stripped = raw_line.strip()
#         if not stripped or stripped.startswith("//") or stripped.startswith("*"):
#             continue
#         for pattern, ctype, severity, cwe, explanation in _CONSTRUCTS:
#             if pattern.search(raw_line):
#                 conf = preproc.get(line_no, 0.0)
#                 detected.append({
#                     "construct_type": ctype,
#                     "line_no":        line_no,
#                     "content":        stripped[:200],
#                     "severity":       severity,
#                     "cwe":            cwe,
#                     "explanation":    explanation,
#                     "preproc_conf":   conf,
#                     "is_flagged":     line_no in susp_set,
#                 })
#
#     detected += _detect_fallthrough(lines)
#
#     # Deduplicate per (line_no, construct_type)
#     best = {}
#     for e in detected:
#         key = (e["line_no"], e["construct_type"])
#         # prefer flagged over non-flagged when deduplicating
#         if key not in best or (e["is_flagged"] and not best[key]["is_flagged"]):
#             best[key] = e
#     detected = sorted(best.values(), key=lambda x: (
#         {"HIGH":3,"MEDIUM":2,"LOW":1}.get(x["severity"],0)), reverse=True)
#
#     # Aggregate counts — these are the features the ML model uses
#     sev_counts = dict(Counter(e["severity"] for e in detected))
#     cwe_counts = dict(Counter(e["cwe"]      for e in detected))
#     n = len(detected); total = max(1, func.get("line_count") or len(lines))
#
#     # Compute overall_syntax_risk as a simple normalised function of counts
#     # so that feature_extractor.py has a meaningful scalar to work with.
#     # The ML model learns what combination of HIGH/MEDIUM/LOW counts predicts vulnerability.
#     high_n   = sev_counts.get("HIGH",   0)
#     medium_n = sev_counts.get("MEDIUM", 0)
#     low_n    = sev_counts.get("LOW",    0)
#     overall  = round(min(1.0, (3*high_n + 2*medium_n + low_n) / max(1, total) * 2.0), 4)
#     dominant_cwe = Counter(e["cwe"] for e in detected).most_common(1)[0][0] if detected else None
#
#     log.info(
#         f"[Task3] {n} constructs  overall_risk={overall:.4f}  "
#         f"severity={sev_counts}  dominant_cwe={dominant_cwe}  strategy={strategy}"
#     )
#
#     return {
#         "task":               "syntax_risk_prediction",
#         "unsafe_constructs":  detected,
#         "overall_syntax_risk":overall,
#         "severity_counts":    sev_counts,
#         "cwe_counts":         cwe_counts,
#         "dominant_cwe":       dominant_cwe,
#         "strategy":           strategy,
#     }
#
#
# def _detect_fallthrough(lines):
#     findings = []; in_case = has_exit = False; case_line = None
#     for line_no, raw_line in enumerate(lines, start=1):
#         if _CASE_PAT.match(raw_line):
#             if in_case and not has_exit and case_line:
#                 findings.append({"construct_type":"implicit_fallthrough","line_no":case_line,
#                     "content":lines[case_line-1].strip()[:200],"severity":"LOW",
#                     "cwe":"CWE-484","explanation":"Case falls through without break/return",
#                     "preproc_conf":0.0,"is_flagged":False})
#             in_case=True; case_line=line_no; has_exit=False
#         elif in_case and _BREAK_PAT.search(raw_line): has_exit=True
#     return findings
#
#
# def _read(path: str) -> Optional[str]:
#     p = Path(path)
#     if not p.exists(): return None
#     try: return p.read_text(encoding="utf-8", errors="replace")
#     except Exception: return None
#
#
# if __name__ == "__main__":
#     import json
#     logging.basicConfig(level=logging.INFO)
#     if len(sys.argv) < 2:
#         print("Usage: python task3_syntax_risk_prediction.py <sample_pred.json>")
#         sys.exit(1)
#     pred = json.loads(Path(sys.argv[1]).read_text())
#     print(json.dumps(run(pred["file"], pred.get("suspicious_lines",[]),
#                          pred.get("func",{}), pred.get("line_map",{})), indent=2))



# =============================================================================
# mtd/task3_syntax_risk_prediction.py  —  Task 3: Syntax Risk Prediction
#
# Detects unsafe constructs and classifies them by severity tier.
# No base severity scores — the ML model learns severity importance.
# Output: counts per severity tier + CWE tags → fed to feature_extractor.
# =============================================================================

import re, logging, math, sys
from collections import Counter
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Detection catalogue — (pattern, construct_type, severity_tier, cwe, explanation)
# No numeric scores — severity is categorical only (HIGH/MEDIUM/LOW)
_CONSTRUCTS = [
    # HIGH
    (re.compile(r"\bgets\s*\("),                  "unbounded_copy",       "HIGH", "CWE-120", "gets() — no bounds"),
    (re.compile(r"\bstrcpy\s*\("),                "unbounded_copy",       "HIGH", "CWE-120", "strcpy() — no bounds"),
    (re.compile(r"\bstrcat\s*\("),                "unbounded_copy",       "HIGH", "CWE-120", "strcat() — no bounds"),
    (re.compile(r"\bsystem\s*\(\s*(?!\s*\")"),    "command_injection",    "HIGH", "CWE-78",  "system() variable arg"),
    (re.compile(r"\bpopen\s*\(\s*(?!\s*\")"),     "command_injection",    "HIGH", "CWE-78",  "popen() variable arg"),
    (re.compile(r"\bexecve?\s*\(\s*(?!\s*\")"),   "command_injection",    "HIGH", "CWE-78",  "exec() variable arg"),
    (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'),     "format_string_inject", "HIGH", "CWE-134", "printf() variable format"),
    (re.compile(r'\bfprintf\s*\(\s*\w+\s*,\s*\w+\s*\)'), "format_string_inject","HIGH","CWE-134","fprintf() variable format"),
    (re.compile(r'\bsyslog\s*\(\s*\w+\s*,\s*\w+\s*\)'),  "format_string_inject","HIGH","CWE-134","syslog() variable format"),
    (re.compile(r'"\.\./'),                       "path_traversal",       "HIGH", "CWE-22",  "hardcoded ../ path traversal"),
    # MEDIUM
    (re.compile(r"\bsprintf\s*\("),               "unbounded_copy",       "MEDIUM","CWE-120","sprintf() — use snprintf"),
    (re.compile(r"\bscanf\s*\("),                 "unbounded_copy",       "MEDIUM","CWE-120","scanf() — add width specifier"),
    (re.compile(r"\bmalloc\s*\([^)]+\)\s*;"),     "unchecked_alloc",      "MEDIUM","CWE-476","malloc() result unchecked"),
    (re.compile(r"\brealloc\s*\("),               "unchecked_alloc",      "MEDIUM","CWE-476","realloc() — original ptr lost"),
    (re.compile(r"\(int\)\s*strlen"),             "integer_truncation",   "MEDIUM","CWE-190","Casting size_t to int"),
    (re.compile(r"\(short\)\s*\w+"),              "integer_truncation",   "MEDIUM","CWE-190","Narrowing cast to short"),
    (re.compile(r"\(char\s*\*\)\s*(?!\"|\()"),    "dangerous_cast",       "MEDIUM","CWE-704","Cast to char* — type safety lost"),
    (re.compile(r"\(void\s*\*\)\s*\w+"),          "dangerous_cast",       "MEDIUM","CWE-704","Cast to void* — loses type info"),
    (re.compile(r"\batoi\s*\("),                  "missing_return_check", "MEDIUM","CWE-190","atoi() error indistinguishable"),
    (re.compile(r"\balloca\s*\("),                "unbounded_copy",       "MEDIUM","CWE-121","alloca() — no size bound"),
    (re.compile(r"^\s*read\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","read() return ignored"),
    (re.compile(r"^\s*write\s*\(", re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","write() return ignored"),
    (re.compile(r"^\s*recv\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","recv() return ignored"),
    (re.compile(r"^\s*send\s*\(",  re.MULTILINE), "missing_return_check", "MEDIUM","CWE-252","send() return ignored"),
    # LOW
    (re.compile(r"\bsnprintf\s*\("),              "unbounded_copy",       "LOW",  "CWE-120","snprintf() — check truncation"),
    (re.compile(r"\bstrncat\s*\("),               "unbounded_copy",       "LOW",  "CWE-120","strncat() — check total length"),
    (re.compile(r"<<\s*\d+"),                     "integer_truncation",   "LOW",  "CWE-190","Bit shift — verify width"),
    (re.compile(r"\w+\+\+\s*\[|\[\s*\w+\+\+\s*\]"),"pointer_arithmetic", "LOW",  "CWE-119","Increment inside array index"),
    (re.compile(r"\brand\s*\("),                  "weak_randomness",      "LOW",  "CWE-330","rand() — not cryptographic"),
    (re.compile(r"\bsrand\s*\(\s*time"),          "weak_randomness",      "LOW",  "CWE-330","srand(time()) — predictable"),
]

_CASE_PAT  = re.compile(r"^\s*case\s+.+:")
_BREAK_PAT = re.compile(r"\b(break|return|goto|continue)\b")


def run(source_file: str, suspicious_lines: list,
        func: dict, line_map: dict) -> dict:
    """Detect unsafe constructs and count by severity. No scores — counts only."""
    code     = func.get("code") or _read(source_file) or ""
    lines    = code.splitlines()
    strategy = line_map.get("strategy", "heuristic")

    preproc = {e["line_no"]: e.get("confidence", 0.5)
               for e in line_map.get("suspicious_lines", []) if e.get("line_no")}
    susp_set = set(suspicious_lines)

    detected = []
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue
        for pattern, ctype, severity, cwe, explanation in _CONSTRUCTS:
            if pattern.search(raw_line):
                conf = preproc.get(line_no, 0.0)
                detected.append({
                    "construct_type": ctype,
                    "line_no":        line_no,
                    "content":        stripped[:200],
                    "severity":       severity,
                    "cwe":            cwe,
                    "explanation":    explanation,
                    "preproc_conf":   conf,
                    "is_flagged":     line_no in susp_set,
                })

    detected += _detect_fallthrough(lines)

    # Deduplicate per (line_no, construct_type)
    best = {}
    for e in detected:
        key = (e["line_no"], e["construct_type"])
        # prefer flagged over non-flagged when deduplicating
        if key not in best or (e["is_flagged"] and not best[key]["is_flagged"]):
            best[key] = e
    detected = sorted(best.values(), key=lambda x: (
        {"HIGH":3,"MEDIUM":2,"LOW":1}.get(x["severity"],0)), reverse=True)

    # Aggregate counts — these are the features the ML model uses
    sev_counts = dict(Counter(e["severity"] for e in detected))
    cwe_counts = dict(Counter(e["cwe"]      for e in detected))
    n = len(detected); total = max(1, func.get("line_count") or len(lines))

    # Load learned severity weights from model JSON.
    # Defaults: w_H=3, w_M=2, w_L=1 (monotonicity: higher severity = higher weight).
    # These are learned by learn_task3_weights() in train.py via correlation
    # maximisation on the validation set, subject to w_H >= w_M >= w_L > 0.
    w_H, w_M, w_L = 3.0, 2.0, 1.0   # safe defaults
    try:
        import json as _json
        from pathlib import Path as _Path
        _mp = (_Path(__file__).resolve().parent
               / "ml" / "models" / "logreg_model.json")
        if _mp.exists():
            _m = _json.loads(_mp.read_text(encoding="utf-8"))
            t3w = _m.get("task3_weights", {})
            w_H = float(t3w.get("w_H", 3.0))
            w_M = float(t3w.get("w_M", 2.0))
            w_L = float(t3w.get("w_L", 1.0))
    except Exception:
        pass   # use defaults if model not available

    high_n   = sev_counts.get("HIGH",   0)
    medium_n = sev_counts.get("MEDIUM", 0)
    low_n    = sev_counts.get("LOW",    0)
    # Normalise by total lines and w_H so score stays in [0, 1]
    overall  = round(
        min(1.0, (w_H * high_n + w_M * medium_n + w_L * low_n)
                 / max(1.0, total * w_H)),
        4
    )
    dominant_cwe = Counter(e["cwe"] for e in detected).most_common(1)[0][0] if detected else None

    log.info(
        f"[Task3] {n} constructs  overall_risk={overall:.4f}  "
        f"severity={sev_counts}  dominant_cwe={dominant_cwe}  strategy={strategy}"
    )

    return {
        "task":               "syntax_risk_prediction",
        "unsafe_constructs":  detected,
        "overall_syntax_risk":overall,
        "severity_counts":    sev_counts,
        "cwe_counts":         cwe_counts,
        "dominant_cwe":       dominant_cwe,
        "strategy":           strategy,
    }


def _detect_fallthrough(lines):
    findings = []; in_case = has_exit = False; case_line = None
    for line_no, raw_line in enumerate(lines, start=1):
        if _CASE_PAT.match(raw_line):
            if in_case and not has_exit and case_line:
                findings.append({"construct_type":"implicit_fallthrough","line_no":case_line,
                    "content":lines[case_line-1].strip()[:200],"severity":"LOW",
                    "cwe":"CWE-484","explanation":"Case falls through without break/return",
                    "preproc_conf":0.0,"is_flagged":False})
            in_case=True; case_line=line_no; has_exit=False
        elif in_case and _BREAK_PAT.search(raw_line): has_exit=True
    return findings


def _read(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists(): return None
    try: return p.read_text(encoding="utf-8", errors="replace")
    except Exception: return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python task3_syntax_risk_prediction.py <sample_pred.json>")
        sys.exit(1)
    pred = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(run(pred["file"], pred.get("suspicious_lines",[]),
                         pred.get("func",{}), pred.get("line_map",{})), indent=2))


