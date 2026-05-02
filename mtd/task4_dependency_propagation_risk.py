# # =============================================================================
# # mtd/task4_dependency_propagation_risk.py  —  Task 4: Dependency Propagation Risk
# #
# # Performs lightweight def-use taint analysis and control-flow analysis.
# # No hardcoded severity scores — raw counts (sources, sinks, paths, nesting)
# # are fed to feature_extractor.py and weighted by the trained ML model.
# # =============================================================================
#
# import re, logging, math, sys
# from pathlib import Path
# from typing import Optional
#
# log = logging.getLogger(__name__)
#
# # Taint source patterns — detection only, no severity scores
# _TAINT_SOURCES = [
#     (re.compile(r"\bgetenv\s*\("),    "getenv"),
#     (re.compile(r"\bfgets?\s*\("),    "file_input"),
#     (re.compile(r"\bscanf\s*\("),     "stdin_scanf"),
#     (re.compile(r"\brecv\s*\("),      "network_recv"),
#     (re.compile(r"\bread\s*\("),      "file_read"),
#     (re.compile(r"\bargv\b"),         "argv"),
#     (re.compile(r"\bmalloc\s*\("),    "heap_alloc"),
#     (re.compile(r"\brealloc\s*\("),   "heap_realloc"),
#     (re.compile(r"\bcalloc\s*\("),    "heap_calloc"),
#     (re.compile(r"\bgetchar\s*\("),   "stdin_char"),
#     (re.compile(r"\bgetline\s*\("),   "getline"),
#     (re.compile(r"\bsscanf\s*\("),    "sscanf"),
#     (re.compile(r"\bstrtol\s*\("),    "strtol"),
#     (re.compile(r"\batoi\s*\("),      "atoi"),
# ]
#
# # Taint sink patterns — detection only
# _TAINT_SINKS = [
#     (re.compile(r"\bstrcpy\s*\("),                "strcpy_dst"),
#     (re.compile(r"\bstrcat\s*\("),                "strcat_dst"),
#     (re.compile(r"\bsprintf\s*\("),               "sprintf_dst"),
#     (re.compile(r"\bmemcpy\s*\("),                "memcpy_dst"),
#     (re.compile(r"\bmemmove\s*\("),               "memmove_dst"),
#     (re.compile(r"\bsystem\s*\("),                "system_cmd"),
#     (re.compile(r"\bpopen\s*\("),                 "popen_cmd"),
#     (re.compile(r"\bexecve?\s*\("),               "exec_cmd"),
#     (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'),    "printf_fmt"),
#     (re.compile(r"\bfopen\s*\("),                 "fopen_path"),
#     (re.compile(r"\bopen\s*\("),                  "open_path"),
#     (re.compile(r"\bfree\s*\("),                  "free_ptr"),
#     (re.compile(r"\w+\s*\[\s*\w+\s*\]"),          "array_index"),
# ]
#
# _ASSIGN       = re.compile(r"\b(\w+)\s*=\s*(.+)")
# _LOOP_KW      = re.compile(r"\b(for|while|do)\b")
# _BRANCH_KW    = re.compile(r"\b(if|else\s+if)\b")
# _NULL_CHECK   = re.compile(r"\b(\w+)\s*(?:==|!=)\s*NULL|\bNULL\s*(?:==|!=)\s*(\w+)")
# _BOUNDS_CHECK = re.compile(r"\b(\w+)\s*[<>]=?\s*\w+|\bsizeof\b")
#
#
# def run(source_file: str, suspicious_lines: list,
#         func: dict, line_map: dict) -> dict:
#     """Analyse data and control flow — raw counts, no hardcoded weights."""
#     code        = func.get("code") or _read(source_file) or ""
#     lines       = code.splitlines()
#     param_count = func.get("param_count", 0) or 0
#     line_count  = func.get("line_count") or len(lines)
#     strategy    = line_map.get("strategy", "heuristic")
#
#     preproc = {}
#     for e in line_map.get("suspicious_lines", []):
#         ln = e.get("line_no")
#         if ln:
#             preproc[ln] = {"confidence": e.get("confidence",0.5), "reason": e.get("reason","")}
#
#     susp_set = set(suspicious_lines)
#
#     data_flow = _analyse_data_flow(lines, preproc, susp_set, param_count)
#     ctrl_flow = _analyse_control_flow(lines, preproc, susp_set, line_count)
#
#     # Compute scalar risk from raw counts — proportional, no hand-tuned weights
#     # ML model learns the right combination from BigVul labels
#     n_sources = len(data_flow["taint_sources"])
#     n_sinks   = len(data_flow["taint_sinks"])
#     n_paths   = len(data_flow["propagation_paths"])
#     nesting   = ctrl_flow["nesting_depth"]
#
#     df_risk = round(min(1.0, (n_sources + n_sinks + 2*n_paths) / max(1, line_count) * 3.0), 4)
#     cf_risk = round(min(1.0, (nesting / 8.0 + len(ctrl_flow["unguarded_branches"]) / max(1, line_count) * 2.0)), 4)
#     overall  = round(0.60 * df_risk + 0.40 * cf_risk, 4)
#
#     log.info(
#         f"[Task4] overall={overall:.4f}  "
#         f"tainted_vars={len(data_flow['tainted_vars'])}  "
#         f"paths={n_paths}  nesting={nesting}  "
#         f"param_count={param_count}  strategy={strategy}"
#     )
#
#     return {
#         "task":                    "dependency_propagation_risk",
#         "data_flow":               {**data_flow, "data_flow_risk": df_risk},
#         "control_flow":            {**ctrl_flow, "control_flow_risk": cf_risk},
#         "overall_dependency_risk": overall,
#         "high_risk_vars":          sorted(set(p["var"] for p in data_flow["propagation_paths"])),
#         "param_count":             param_count,
#         "strategy":                strategy,
#     }
#
#
# def _analyse_data_flow(lines, preproc, susp_set, param_count):
#     sources = []; tainted = {}
#
#     # Function parameters are external input — treat as taint sources
#     if param_count > 0 and lines:
#         sig = " ".join(lines[:min(6, len(lines))])
#         skip = {"void","int","char","long","short","unsigned","signed",
#                 "const","static","inline","struct","enum","return",
#                 "if","for","while","do","switch","else","NULL"}
#         tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", sig)
#         seen = set()
#         for i, tok in enumerate(tokens):
#             if tok in skip or tok in seen: continue
#             # Check if this token is a parameter (preceded by type keyword)
#             if i > 0 and tokens[i-1] in skip | {"*"}:
#                 seen.add(tok)
#                 tainted[tok] = min(1.0, 0.5 + param_count * 0.05)
#                 sources.append({"line_no":1,"var":tok,"source_label":"function_param",
#                                  "preproc_conf":0.5,"content":sig[:80]})
#
#     for line_no, raw_line in enumerate(lines, start=1):
#         conf = preproc.get(line_no, {}).get("confidence", 0.0)
#         for pat, label in _TAINT_SOURCES:
#             if pat.search(raw_line):
#                 var = _lhs(raw_line)
#                 sources.append({"line_no":line_no,"var":var or "?","source_label":label,
#                                  "preproc_conf":conf,"content":raw_line.strip()[:120]})
#                 if var: tainted[var] = max(tainted.get(var, 0.0), 0.5 + 0.5*conf)
#
#     sinks = []
#     for line_no, raw_line in enumerate(lines, start=1):
#         conf = preproc.get(line_no, {}).get("confidence", 0.0)
#         for pat, label in _TAINT_SINKS:
#             if pat.search(raw_line):
#                 involved = [v for v in tainted if re.search(r"\b"+re.escape(v)+r"\b", raw_line)]
#                 sinks.append({"line_no":line_no,"sink_label":label,"content":raw_line.strip()[:120],
#                                "tainted_vars_present":involved,"preproc_conf":conf,
#                                "is_flagged":line_no in susp_set})
#
#     paths = []
#     for sink in sinks:
#         for var in sink["tainted_vars_present"]:
#             paths.append({"var":var,"sink_label":sink["sink_label"],
#                           "sink_line":sink["line_no"],"flagged":sink["is_flagged"]})
#
#     return {"tainted_vars":sorted(tainted.keys()),"taint_sources":sources,
#             "taint_sinks":sinks,"propagation_paths":paths}
# def _analyse_control_flow(lines, preproc, susp_set, line_count):
#     loop_count=0; nesting=0; depth=0; unguarded=[]; last_guard=False
#     for line_no, raw_line in enumerate(lines, start=1):
#         stripped = raw_line.strip()
#         conf = preproc.get(line_no, {}).get("confidence", 0.0)
#         depth += stripped.count("{") - stripped.count("}")
#         nesting = max(nesting, depth)
#         if _LOOP_KW.search(raw_line):
#             loop_count += 1
#             if re.search(r"\w+\s*\[\s*\w+\s*\]", raw_line) and not _BOUNDS_CHECK.search(raw_line):
#                 unguarded.append({"line_no":line_no,"content":stripped[:120],
#                                    "reason":"array access in loop without bounds check","preproc_conf":conf})
#         if _BRANCH_KW.search(raw_line):
#             last_guard = bool(_NULL_CHECK.search(raw_line) or _BOUNDS_CHECK.search(raw_line))
#         elif not last_guard:
#             if any(re.search(p, raw_line) for p in [r"\bstrcpy\b",r"\bmemcpy\b",r"\bfree\b",r"\bsystem\b"]):
#                 if stripped and not stripped.startswith("//"):
#                     unguarded.append({"line_no":line_no,"content":stripped[:120],
#                                        "reason":"risky op without NULL/bounds guard","preproc_conf":conf})
#     return {"loop_count":loop_count,"nesting_depth":nesting,"unguarded_branches":unguarded}
#
#
# def _lhs(line: str) -> Optional[str]:
#     m = _ASSIGN.search(line)
#     if m:
#         lhs = m.group(1).strip()
#         if lhs and lhs.isidentifier() and lhs not in {"if","for","while","return","else","switch"}:
#             return lhs
#     return None
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
#         print("Usage: python task4_dependency_propagation_risk.py <sample_pred.json>")
#         sys.exit(1)
#     pred = json.loads(Path(sys.argv[1]).read_text())
#     print(json.dumps(run(pred["file"], pred.get("suspicious_lines",[]),
#                          pred.get("func",{}), pred.get("line_map",{})), indent=2))
#
#


# =============================================================================
# mtd/task4_dependency_propagation_risk.py  —  Task 4: Dependency Propagation Risk
#
# Performs lightweight def-use taint analysis and control-flow analysis.
# No hardcoded severity scores — raw counts (sources, sinks, paths, nesting)
# are fed to feature_extractor.py and weighted by the trained ML model.
# =============================================================================

import re, logging, math, sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Taint source patterns — detection only, no severity scores
_TAINT_SOURCES = [
    (re.compile(r"\bgetenv\s*\("),    "getenv"),
    (re.compile(r"\bfgets?\s*\("),    "file_input"),
    (re.compile(r"\bscanf\s*\("),     "stdin_scanf"),
    (re.compile(r"\brecv\s*\("),      "network_recv"),
    (re.compile(r"\bread\s*\("),      "file_read"),
    (re.compile(r"\bargv\b"),         "argv"),
    (re.compile(r"\bmalloc\s*\("),    "heap_alloc"),
    (re.compile(r"\brealloc\s*\("),   "heap_realloc"),
    (re.compile(r"\bcalloc\s*\("),    "heap_calloc"),
    (re.compile(r"\bgetchar\s*\("),   "stdin_char"),
    (re.compile(r"\bgetline\s*\("),   "getline"),
    (re.compile(r"\bsscanf\s*\("),    "sscanf"),
    (re.compile(r"\bstrtol\s*\("),    "strtol"),
    (re.compile(r"\batoi\s*\("),      "atoi"),
]

# Taint sink patterns — detection only
_TAINT_SINKS = [
    (re.compile(r"\bstrcpy\s*\("),                "strcpy_dst"),
    (re.compile(r"\bstrcat\s*\("),                "strcat_dst"),
    (re.compile(r"\bsprintf\s*\("),               "sprintf_dst"),
    (re.compile(r"\bmemcpy\s*\("),                "memcpy_dst"),
    (re.compile(r"\bmemmove\s*\("),               "memmove_dst"),
    (re.compile(r"\bsystem\s*\("),                "system_cmd"),
    (re.compile(r"\bpopen\s*\("),                 "popen_cmd"),
    (re.compile(r"\bexecve?\s*\("),               "exec_cmd"),
    (re.compile(r'\bprintf\s*\(\s*\w+\s*\)'),    "printf_fmt"),
    (re.compile(r"\bfopen\s*\("),                 "fopen_path"),
    (re.compile(r"\bopen\s*\("),                  "open_path"),
    (re.compile(r"\bfree\s*\("),                  "free_ptr"),
    (re.compile(r"\w+\s*\[\s*\w+\s*\]"),          "array_index"),
]

_ASSIGN       = re.compile(r"\b(\w+)\s*=\s*(.+)")
_LOOP_KW      = re.compile(r"\b(for|while|do)\b")
_BRANCH_KW    = re.compile(r"\b(if|else\s+if)\b")
_NULL_CHECK   = re.compile(r"\b(\w+)\s*(?:==|!=)\s*NULL|\bNULL\s*(?:==|!=)\s*(\w+)")
_BOUNDS_CHECK = re.compile(r"\b(\w+)\s*[<>]=?\s*\w+|\bsizeof\b")


def run(source_file: str, suspicious_lines: list,
        func: dict, line_map: dict) -> dict:
    """Analyse data and control flow — raw counts, no hardcoded weights."""
    code        = func.get("code") or _read(source_file) or ""
    lines       = code.splitlines()
    param_count = func.get("param_count", 0) or 0
    line_count  = func.get("line_count") or len(lines)
    strategy    = line_map.get("strategy", "heuristic")

    preproc = {}
    for e in line_map.get("suspicious_lines", []):
        ln = e.get("line_no")
        if ln:
            preproc[ln] = {"confidence": e.get("confidence",0.5), "reason": e.get("reason","")}

    susp_set = set(suspicious_lines)

    data_flow = _analyse_data_flow(lines, preproc, susp_set, param_count)
    ctrl_flow = _analyse_control_flow(lines, preproc, susp_set, line_count)

    # Compute scalar risk from raw counts — proportional, no hand-tuned weights
    # ML model learns the right combination from BigVul labels
    n_sources = len(data_flow["taint_sources"])
    n_sinks   = len(data_flow["taint_sinks"])
    n_paths   = len(data_flow["propagation_paths"])
    nesting   = ctrl_flow["nesting_depth"]

    df_risk = round(min(1.0, (n_sources + n_sinks + 2*n_paths) / max(1, line_count) * 3.0), 4)
    cf_risk = round(min(1.0, (nesting / 8.0 + len(ctrl_flow["unguarded_branches"]) / max(1, line_count) * 2.0)), 4)

    # Load learned blend weights from model JSON.
    # w_df = weight for data-flow risk, w_cf = weight for control-flow risk
    # w_df + w_cf = 1, both learned by learn_task4_weights() in train.py
    # via Pearson correlation maximisation on the validation set.
    # Defaults: w_df=0.60, w_cf=0.40 (data-flow dominates in most memory-safety bugs)
    w_df, w_cf = 0.60, 0.40
    try:
        import json as _json
        from pathlib import Path as _Path
        _mp = (_Path(__file__).resolve().parent
               / "ml" / "models" / "logreg_model.json")
        if _mp.exists():
            _m  = _json.loads(_mp.read_text(encoding="utf-8"))
            t4w = _m.get("task4_weights", {})
            w_df = float(t4w.get("w_df", 0.60))
            w_cf = float(t4w.get("w_cf", 0.40))
    except Exception:
        pass   # use defaults if model not available

    overall  = round(w_df * df_risk + w_cf * cf_risk, 4)

    log.info(
        f"[Task4] overall={overall:.4f}  "
        f"tainted_vars={len(data_flow['tainted_vars'])}  "
        f"paths={n_paths}  nesting={nesting}  "
        f"param_count={param_count}  strategy={strategy}"
    )

    return {
        "task":                    "dependency_propagation_risk",
        "data_flow":               {**data_flow, "data_flow_risk": df_risk},
        "control_flow":            {**ctrl_flow, "control_flow_risk": cf_risk},
        "overall_dependency_risk": overall,
        "high_risk_vars":          sorted(set(p["var"] for p in data_flow["propagation_paths"])),
        "param_count":             param_count,
        "strategy":                strategy,
    }


def _analyse_data_flow(lines, preproc, susp_set, param_count):
    sources = []; tainted = {}

    # Function parameters are external input — treat as taint sources
    if param_count > 0 and lines:
        sig = " ".join(lines[:min(6, len(lines))])
        skip = {"void","int","char","long","short","unsigned","signed",
                "const","static","inline","struct","enum","return",
                "if","for","while","do","switch","else","NULL"}
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", sig)
        seen = set()
        for i, tok in enumerate(tokens):
            if tok in skip or tok in seen: continue
            # Check if this token is a parameter (preceded by type keyword)
            if i > 0 and tokens[i-1] in skip | {"*"}:
                seen.add(tok)
                tainted[tok] = min(1.0, 0.5 + param_count * 0.05)
                sources.append({"line_no":1,"var":tok,"source_label":"function_param",
                                 "preproc_conf":0.5,"content":sig[:80]})

    for line_no, raw_line in enumerate(lines, start=1):
        conf = preproc.get(line_no, {}).get("confidence", 0.0)
        for pat, label in _TAINT_SOURCES:
            if pat.search(raw_line):
                var = _lhs(raw_line)
                sources.append({"line_no":line_no,"var":var or "?","source_label":label,
                                 "preproc_conf":conf,"content":raw_line.strip()[:120]})
                if var: tainted[var] = max(tainted.get(var, 0.0), 0.5 + 0.5*conf)

    sinks = []
    for line_no, raw_line in enumerate(lines, start=1):
        conf = preproc.get(line_no, {}).get("confidence", 0.0)
        for pat, label in _TAINT_SINKS:
            if pat.search(raw_line):
                involved = [v for v in tainted if re.search(r"\b"+re.escape(v)+r"\b", raw_line)]
                sinks.append({"line_no":line_no,"sink_label":label,"content":raw_line.strip()[:120],
                               "tainted_vars_present":involved,"preproc_conf":conf,
                               "is_flagged":line_no in susp_set})

    paths = []
    for sink in sinks:
        for var in sink["tainted_vars_present"]:
            paths.append({"var":var,"sink_label":sink["sink_label"],
                          "sink_line":sink["line_no"],"flagged":sink["is_flagged"]})

    return {"tainted_vars":sorted(tainted.keys()),"taint_sources":sources,
            "taint_sinks":sinks,"propagation_paths":paths}
def _analyse_control_flow(lines, preproc, susp_set, line_count):
    loop_count=0; nesting=0; depth=0; unguarded=[]; last_guard=False
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        conf = preproc.get(line_no, {}).get("confidence", 0.0)
        depth += stripped.count("{") - stripped.count("}")
        nesting = max(nesting, depth)
        if _LOOP_KW.search(raw_line):
            loop_count += 1
            if re.search(r"\w+\s*\[\s*\w+\s*\]", raw_line) and not _BOUNDS_CHECK.search(raw_line):
                unguarded.append({"line_no":line_no,"content":stripped[:120],
                                   "reason":"array access in loop without bounds check","preproc_conf":conf})
        if _BRANCH_KW.search(raw_line):
            last_guard = bool(_NULL_CHECK.search(raw_line) or _BOUNDS_CHECK.search(raw_line))
        elif not last_guard:
            if any(re.search(p, raw_line) for p in [r"\bstrcpy\b",r"\bmemcpy\b",r"\bfree\b",r"\bsystem\b"]):
                if stripped and not stripped.startswith("//"):
                    unguarded.append({"line_no":line_no,"content":stripped[:120],
                                       "reason":"risky op without NULL/bounds guard","preproc_conf":conf})
    return {"loop_count":loop_count,"nesting_depth":nesting,"unguarded_branches":unguarded}


def _lhs(line: str) -> Optional[str]:
    m = _ASSIGN.search(line)
    if m:
        lhs = m.group(1).strip()
        if lhs and lhs.isidentifier() and lhs not in {"if","for","while","return","else","switch"}:
            return lhs
    return None


def _read(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists(): return None
    try: return p.read_text(encoding="utf-8", errors="replace")
    except Exception: return None


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python task4_dependency_propagation_risk.py <sample_pred.json>")
        sys.exit(1)
    pred = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(run(pred["file"], pred.get("suspicious_lines",[]),
                         pred.get("func",{}), pred.get("line_map",{})), indent=2))


