# # diagnosis/run_diagnosis.py
# import json
# from pathlib import Path
# import sys
#
# # Add project root to sys.path for robust imports
# CURRENT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = CURRENT_DIR.parent
# sys.path.append(str(PROJECT_ROOT))
#
# # Imports
# try:
#     from diagnosis.causal_probe import causal_effects
#     from diagnosis.pdg_utils import pdg_reachable
# except ImportError:
#     from causal_probe import causal_effects
#     from pdg_utils import pdg_reachable
#
# from mtd.syntax_detector import contains_dangerous_api   # adjust if needed
#
# MTD_PATH = Path("/mnt/data/junaid/linevul/linevul/outputs/sample_for_mtd.json")
# OUT_PATH = Path("/mnt/data/junaid/linevul/linevul/outputs/diagnosis_report.json")
#
# data = json.load(open(MTD_PATH))
# susp_lines = data["pred"]["suspicious_lines"]
# texts = data["suspicious_texts"]
# full_code = data["full_code"]  # list, index 1 = line 1
#
# results = []
#
# for ln in susp_lines:
#     key = str(ln)
#     if key not in texts:
#         print(f"[DIAG] Skipping line {ln} → no text available")
#         continue
#
#     line = texts[key]
#
#     # Get non-benign suspicious lines based on syntax
#     non_benign = [
#         v for v in susp_lines
#         if v != ln and contains_dangerous_api(texts.get(str(v), ""))
#     ]
#
#     # Check PDG connectivity
#     connected = any(
#         pdg_reachable(ln, nb) or pdg_reachable(nb, ln)
#         for nb in non_benign
#     ) if non_benign else False
#
#     # Get all perturbation effects
#     effects = causal_effects(full_code, ln, line)
#
#     for e in effects:
#         # Your preferred condition:
#         # Include if:
#         #   - not connected AND delta > very small threshold   OR
#         #   - it's one of the specially interesting perturbation types
#         if (not connected and e["delta"] > 0.05) or e["type"] in ["comment_out", "dead_danger"]:
#             results.append({
#                 "line": ln,
#                 "code": line.strip(),
#                 "feature": e["feature"],
#                 "type": e["type"],
#                 "delta": round(e["delta"], 6),
#                 "base_benign_prob": round(e["base"], 4),
#                 "new_benign_prob": round(e["new"], 4),
#                 "example": e["code"][:120] + ("..." if len(e["code"]) > 120 else ""),
#                 "significance": "high" if e["delta"] > 0.03 else "low/zero"
#             })
#
# # Optional: better sorting - prioritize interesting types first, then by delta
# priority_order = {"dead_danger": 3, "comment_out": 2, "api_mask": 1, "var_rename": 0}
# results.sort(key=lambda x: (-priority_order.get(x["type"], 0), -x["delta"]))
#
# # Save result
# with open(OUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2)
#
# print(f"Diagnosis complete → {len(results)} items found")
# print(f"Report saved to: {OUT_PATH}")
#
# if not results:
#     print("→ No items matched the criteria (very rare with this condition)")









# =============================================================================
# diagnosis/run_diagnosis.py  —  Diagnosis Module
#
# FRAMEWORK POSITION:
#   Preprocessing → MTD → ALC → [DIAGNOSIS] → Repair
#                                    ↑
#                         Triggered only when ALC = UNTRUSTWORTHY
#
# PURPOSE:
#   When ALC flags a result as UNTRUSTWORTHY, this module performs three
#   interconnected analyses to explain WHY the four tasks disagreed:
#
#   Stage 1 — Root Cause Analysis (RCA)
#       Identifies WHICH task is the outlier and WHY it disagrees with the rest.
#       Produces a ranked list of conflict sources with explanations.
#
#   Stage 2 — Vulnerability Context Analysis (VCA)
#       Examines the structural and semantic context of the function to
#       explain what kind of code pattern is causing the disagreement.
#       Maps CWE tags, dangerous constructs, and taint flows to specific lines.
#
#   Stage 3 — Error Source Detection (ESD)
#       Determines whether the disagreement is caused by:
#         (a) a genuine vulnerability that some tasks missed   [MISSED_VULN]
#         (b) a false pattern match inflating one task score   [FALSE_PATTERN]
#         (c) strategy quality — the suspicious-line map is noisy  [NOISY_STRATEGY]
#         (d) a borderline function that is genuinely ambiguous    [AMBIGUOUS]
#
# INPUTS  (from ALC output directory):
#   alc_result.json   — full ALC three-stage report (written by alc/run_alc.py)
#   mtd_result.json   — full MTD four-task report   (written by mtd/run_mtd.py)
#
# OUTPUTS:
#   diagnosis_result.json  — full three-stage diagnosis report
#   diagnosis_summary.txt  — human-readable one-page summary
# =============================================================================

import argparse
import json
import logging
import math
import sys
from pathlib import Path

DIAG_DIR = Path(__file__).resolve().parent
ROOT     = DIAG_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── CWE severity catalogue (for context enrichment) ──────────────────────────
_CWE_DESCRIPTIONS = {
    "CWE-120": "Buffer copy without checking size of input (Classic Buffer Overflow)",
    "CWE-121": "Stack-based buffer overflow",
    "CWE-122": "Heap-based buffer overflow",
    "CWE-119": "Improper restriction of operations within bounds of memory buffer",
    "CWE-125": "Out-of-bounds read",
    "CWE-190": "Integer overflow or wraparound",
    "CWE-191": "Integer underflow",
    "CWE-416": "Use after free",
    "CWE-476": "NULL pointer dereference",
    "CWE-78":  "OS command injection",
    "CWE-134": "Uncontrolled format string",
    "CWE-22":  "Path traversal",
    "CWE-330": "Use of insufficiently random values",
    "CWE-252": "Unchecked return value",
    "CWE-704": "Incorrect type conversion or cast",
    "CWE-484": "Switch statement fall-through",
    "struct-array": "Array indexing with variable index (potential OOB)",
    "struct-cast":  "Unsafe struct/pointer cast",
    "struct-ptr":   "Pointer arithmetic",
}

_CWE_SEVERITY = {
    "CWE-120": "HIGH", "CWE-121": "HIGH", "CWE-122": "HIGH",
    "CWE-78":  "HIGH", "CWE-134": "HIGH", "CWE-416": "HIGH",
    "CWE-190": "MEDIUM", "CWE-476": "MEDIUM", "CWE-119": "MEDIUM",
    "CWE-704": "MEDIUM", "CWE-252": "MEDIUM", "CWE-22": "MEDIUM",
    "CWE-330": "LOW", "CWE-484": "LOW", "CWE-191": "LOW",
    "struct-array": "LOW", "struct-cast": "MEDIUM", "struct-ptr": "LOW",
}


# =============================================================================
# Stage 1 — Root Cause Analysis
# =============================================================================

def root_cause_analysis(task_scores: dict, task_results: dict,
                        alc_stage1: dict, V: float) -> dict:
    """
    Identify WHICH task caused the cross-task conflict and WHY.

    Steps:
      1. Find the outlier task (furthest from mean)
      2. Categorise the conflict pattern
      3. Generate natural-language explanations
      4. Rank conflict pairs by severity
    """
    log.info("--- Diagnosis Stage 1: Root Cause Analysis ---")

    scores     = task_scores
    vals       = list(scores.values())
    task_names = list(scores.keys())
    n          = len(vals)
    mean_s     = sum(vals) / n
    variance   = sum((v - mean_s) ** 2 for v in vals) / n

    # Identify outlier task
    deviations = {k: abs(v - mean_s) for k, v in scores.items()}
    outlier    = max(deviations, key=deviations.get)
    outlier_score = scores[outlier]
    outlier_direction = "HIGH" if outlier_score > mean_s else "LOW"

    # Conflict pattern classification
    high_tasks = [k for k, v in scores.items() if v > 0.5]
    low_tasks  = [k for k, v in scores.items() if v <= 0.5]

    if len(high_tasks) == 1:
        pattern = "SINGLE_HIGH_OUTLIER"
        pattern_desc = (
            f"Only {high_tasks[0]} reports high risk ({scores[high_tasks[0]]:.3f}) "
            f"while all other tasks report low risk. This single-task flag is "
            f"the primary source of uncertainty."
        )
    elif len(high_tasks) == 3 and len(low_tasks) == 1:
        pattern = "SINGLE_LOW_OUTLIER"
        pattern_desc = (
            f"Three tasks report elevated risk but {low_tasks[0]} reports low risk "
            f"({scores[low_tasks[0]]:.3f}). The missing signal in {low_tasks[0]} "
            f"creates the inconsistency."
        )
    elif len(high_tasks) == 2:
        pattern = "SPLIT_DISAGREEMENT"
        pattern_desc = (
            f"Tasks are evenly split: {high_tasks} report high risk while "
            f"{low_tasks} report low risk. This suggests the function has "
            f"genuine ambiguity — some risk dimensions present, others absent."
        )
    elif len(high_tasks) == 0:
        pattern = "ALL_LOW_MARGINAL"
        pattern_desc = (
            "All tasks report low risk but the overall V score is uncertain. "
            "This may be caused by subtle pattern hits aggregating to a borderline score."
        )
    else:
        pattern = "MULTI_CONFLICT"
        pattern_desc = (
            "Multiple tasks show conflicting signals without a clear majority. "
            "The function likely contains multiple independent risk factors "
            "that pull in different directions."
        )

    # Per-task root cause explanation
    task_explanations = {}
    r1 = task_results.get("task1", {})
    r2 = task_results.get("task2", {})
    r3 = task_results.get("task3", {})
    r4 = task_results.get("task4", {})

    # Task 1 explanation
    hits    = r1.get("features", {}).get("pattern_hit_count", 0)
    cwes    = r1.get("features", {}).get("cwe_tags", [])
    s1      = scores.get("task1", 0)
    if s1 > 0.3:
        task_explanations["task1"] = (
            f"Pattern classifier flagged {hits} pattern hit(s) matching "
            f"{cwes if cwes else 'generic patterns'}. Score={s1:.3f} indicates "
            f"moderate-to-high pattern density."
        )
    elif s1 > 0:
        task_explanations["task1"] = (
            f"Pattern classifier detected {hits} hit(s) but low density. "
            f"CWEs detected: {cwes if cwes else 'none'}. Score={s1:.3f}."
        )
    else:
        task_explanations["task1"] = (
            f"Pattern classifier found no matching vulnerability patterns. "
            f"Score=0.000 — function has no known dangerous API calls or constructs."
        )

    # Task 2 explanation
    risky_count = r2.get("summary", {}).get("risky_line_count", 0)
    total_lines = r2.get("summary", {}).get("total_lines", 1)
    top_line    = r2.get("top_line")
    s2          = scores.get("task2", 0)
    ratio       = risky_count / max(1, total_lines)
    if s2 > 0.7:
        task_explanations["task2"] = (
            f"Line localizer flagged {risky_count}/{total_lines} lines as risky "
            f"(ratio={ratio:.2f}). Top risky line: {top_line}. HIGH score={s2:.3f} "
            f"is driven by the heuristic line mapper finding pointer/array operations "
            f"throughout the function — even in safe code this is common."
        )
    elif s2 > 0.3:
        task_explanations["task2"] = (
            f"Line localizer found {risky_count} risky lines. Score={s2:.3f} "
            f"suggests moderate structural risk indicators present."
        )
    else:
        task_explanations["task2"] = (
            f"Line localizer found few or no risky lines ({risky_count}). "
            f"Score={s2:.3f} — function has minimal structural risk markers."
        )

    # Task 3 explanation
    constructs  = r3.get("unsafe_constructs", [])
    sev_counts  = r3.get("severity_counts", {})
    dom_cwe     = r3.get("dominant_cwe")
    s3          = scores.get("task3", 0)
    if s3 > 0.3:
        task_explanations["task3"] = (
            f"Syntax analyser found {len(constructs)} unsafe construct(s): "
            f"{sev_counts}. Dominant CWE: {dom_cwe}. Score={s3:.3f}."
        )
    elif s3 > 0:
        task_explanations["task3"] = (
            f"Syntax analyser found {len(constructs)} construct(s) with low severity. "
            f"Score={s3:.3f} — constructs present but not high-risk."
        )
    else:
        task_explanations["task3"] = (
            f"Syntax analyser found no unsafe constructs. Score=0.000 — "
            f"function uses no unsafe APIs or dangerous coding patterns "
            f"detectable by static syntax analysis."
        )

    # Task 4 explanation
    df          = r4.get("data_flow",    {})
    cf          = r4.get("control_flow", {})
    tainted     = df.get("tainted_vars", [])
    paths       = df.get("propagation_paths", [])
    nesting     = cf.get("nesting_depth", 0)
    s4          = scores.get("task4", 0)
    if s4 > 0.5:
        task_explanations["task4"] = (
            f"Dependency analyser found {len(tainted)} tainted variable(s), "
            f"{len(paths)} propagation path(s), nesting depth={nesting}. "
            f"Score={s4:.3f} — significant data flow risk detected."
        )
    elif s4 > 0.2:
        task_explanations["task4"] = (
            f"Dependency analyser found {len(tainted)} tainted variable(s). "
            f"Score={s4:.3f} — moderate taint propagation present."
        )
    else:
        task_explanations["task4"] = (
            f"Dependency analyser found minimal taint flow. "
            f"Score={s4:.3f} — function has limited data dependency risk."
        )

    # Conflict pair ranking
    conflict_pairs = alc_stage1.get("pairwise_pairs", [])
    conflicting    = [p for p in conflict_pairs if p.get("conflict")]
    conflicting.sort(key=lambda x: x.get("diff", 0), reverse=True)

    log.info(
        f"[Diagnosis/RCA] outlier={outlier}({outlier_direction})  "
        f"pattern={pattern}  conflicts={len(conflicting)}/{len(conflict_pairs)}  "
        f"variance={variance:.4f}"
    )

    return {
        "outlier_task":        outlier,
        "outlier_score":       round(outlier_score, 4),
        "outlier_direction":   outlier_direction,
        "mean_task_score":     round(mean_s, 4),
        "variance":            round(variance, 4),
        "conflict_pattern":    pattern,
        "pattern_description": pattern_desc,
        "task_explanations":   task_explanations,
        "top_conflict_pairs":  conflicting[:3],
        "high_risk_tasks":     high_tasks,
        "low_risk_tasks":      low_tasks,
    }


# =============================================================================
# Stage 2 — Vulnerability Context Analysis
# =============================================================================

def vulnerability_context_analysis(task_results: dict, mtd_result: dict,
                                    V: float, strategy: str) -> dict:
    """
    Examine the structural and semantic context of the function.
    Maps CWE tags, dangerous constructs, and taint flows to specific lines.
    Explains WHAT kind of code is triggering the disagreement.
    """
    log.info("--- Diagnosis Stage 2: Vulnerability Context Analysis ---")

    r1 = task_results.get("task1", {})
    r2 = task_results.get("task2", {})
    r3 = task_results.get("task3", {})
    r4 = task_results.get("task4", {})

    func_lines  = mtd_result.get("func_lines", 0)
    func_name   = mtd_result.get("func_name", "unknown")

    # Collect all CWE tags from all tasks
    cwe_tags_t1 = set(r1.get("features", {}).get("cwe_tags", []))
    cwe_tags_t3 = set(c.get("cwe", "") for c in r3.get("unsafe_constructs", []))
    all_cwes    = cwe_tags_t1 | cwe_tags_t3

    # Build CWE context with descriptions and severity
    cwe_context = {}
    for cwe in all_cwes:
        if cwe:
            cwe_context[cwe] = {
                "description": _CWE_DESCRIPTIONS.get(cwe, "Unknown vulnerability pattern"),
                "severity":    _CWE_SEVERITY.get(cwe, "UNKNOWN"),
                "detected_by": [],
            }
            if cwe in cwe_tags_t1:
                cwe_context[cwe]["detected_by"].append("task1_pattern_classifier")
            if cwe in cwe_tags_t3:
                cwe_context[cwe]["detected_by"].append("task3_syntax_analyser")

    # Highest severity CWE found
    severity_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
    dominant_cwe   = None
    dom_severity   = "UNKNOWN"
    for cwe, info in cwe_context.items():
        if severity_order.get(info["severity"], 0) > severity_order.get(dom_severity, 0):
            dominant_cwe = cwe
            dom_severity = info["severity"]

    # Dangerous construct summary from Task 3
    constructs = r3.get("unsafe_constructs", [])
    construct_summary = []
    for c in constructs[:5]:   # top 5
        construct_summary.append({
            "line":        c.get("line_no"),
            "type":        c.get("construct_type"),
            "severity":    c.get("severity"),
            "cwe":         c.get("cwe"),
            "explanation": c.get("explanation"),
            "content":     c.get("content", "")[:80],
        })

    # Taint flow summary from Task 4
    df      = r4.get("data_flow", {})
    sources = df.get("taint_sources", [])[:5]
    sinks   = df.get("taint_sinks",   [])[:5]
    paths   = df.get("propagation_paths", [])[:5]

    taint_summary = {
        "tainted_variables": df.get("tainted_vars", []),
        "source_count":      len(df.get("taint_sources", [])),
        "sink_count":        len(df.get("taint_sinks", [])),
        "path_count":        len(df.get("propagation_paths", [])),
        "top_sources":       [{"line": s.get("line_no"), "label": s.get("source_label"),
                               "content": s.get("content", "")[:60]} for s in sources],
        "top_sinks":         [{"line": s.get("line_no"), "label": s.get("sink_label"),
                               "content": s.get("content", "")[:60]} for s in sinks],
        "propagation_paths": [{"var": p.get("var"), "sink": p.get("sink_label"),
                               "line": p.get("sink_line")} for p in paths],
    }

    # Risky line map from Task 2
    risky_lines = r2.get("risky_lines", [])[:10]
    risky_map   = [{"line": l.get("line_no"), "score": l.get("risk_score"),
                    "reasons": l.get("reasons", []),
                    "content": l.get("content", "")[:60]} for l in risky_lines]

    # Strategy quality assessment
    strategy_quality_map = {"ground_truth": 1.0, "heuristic": 0.8, "all_lines": 0.5}
    strat_quality = strategy_quality_map.get(strategy, 0.7)
    if strategy == "ground_truth":
        strat_note = "Suspicious lines come from human CVE annotation — highest confidence."
    elif strategy == "heuristic":
        strat_note = (
            "Suspicious lines come from pattern-based heuristics. "
            "Line mapper may flag safe array/pointer operations as suspicious, "
            "inflating Task 2 scores even in clean functions."
        )
    else:
        strat_note = (
            "All lines treated as equally suspicious (all_lines strategy). "
            "Task 2 will flag risky structural patterns throughout the entire function. "
            "This reduces precision and is the most common source of Task 2 inflation."
        )

    # Function complexity indicators
    cf          = r4.get("control_flow", {})
    nesting     = cf.get("nesting_depth", 0)
    loop_count  = cf.get("loop_count",    0)
    param_count = r4.get("param_count",   0)

    complexity_note = ""
    if func_lines > 100:
        complexity_note += f"Large function ({func_lines} lines) increases false-positive risk. "
    if nesting > 4:
        complexity_note += f"Deep nesting (depth={nesting}) suggests complex control flow. "
    if param_count > 4:
        complexity_note += f"Many parameters ({param_count}) increase taint surface. "
    if loop_count > 3:
        complexity_note += f"Multiple loops ({loop_count}) may cause Task 2 inflation. "
    if not complexity_note:
        complexity_note = "Function complexity is within normal bounds."

    log.info(
        f"[Diagnosis/VCA] cwes={sorted(all_cwes)}  dominant={dominant_cwe}({dom_severity})  "
        f"constructs={len(constructs)}  taint_paths={len(paths)}  "
        f"strategy={strategy}  lines={func_lines}"
    )

    return {
        "function_name":       func_name,
        "function_lines":      func_lines,
        "strategy":            strategy,
        "strategy_quality":    strat_quality,
        "strategy_note":       strat_note,
        "cwe_context":         cwe_context,
        "dominant_cwe":        dominant_cwe,
        "dominant_severity":   dom_severity,
        "dangerous_constructs":construct_summary,
        "taint_flow":          taint_summary,
        "risky_line_map":      risky_map,
        "complexity_note":     complexity_note,
    }


# =============================================================================
# Stage 3 — Error Source Detection
# =============================================================================

def error_source_detection(rca: dict, vca: dict,
                            task_scores: dict, V: float,
                            strategy: str, mtd_verdict: str) -> dict:
    """
    Determine the root ERROR SOURCE — why did the four tasks disagree?

    Error source categories:
      MISSED_VULN      — genuine vulnerability that some tasks failed to detect
      FALSE_PATTERN    — a pattern match inflated one task's score artificially
      NOISY_STRATEGY   — the suspicious-line strategy is too noisy for this function
      AMBIGUOUS        — function is genuinely borderline (near-threshold V)

    Each category implies a different repair action.
    """
    log.info("--- Diagnosis Stage 3: Error Source Detection ---")

    s1 = task_scores.get("task1", 0)
    s2 = task_scores.get("task2", 0)
    s3 = task_scores.get("task3", 0)
    s4 = task_scores.get("task4", 0)

    outlier       = rca["outlier_task"]
    conflict_pat  = rca["conflict_pattern"]
    high_tasks    = rca["high_risk_tasks"]
    cwe_context   = vca["cwe_context"]
    dom_severity  = vca["dominant_severity"]
    taint_paths   = vca["taint_flow"]["path_count"]
    constructs    = vca["dangerous_constructs"]
    strat_quality = vca["strategy_quality"]
    func_lines    = vca["function_lines"]

    evidence = []
    scores_list = [("MISSED_VULN", 0), ("FALSE_PATTERN", 0),
                   ("NOISY_STRATEGY", 0), ("AMBIGUOUS", 0)]
    scores_dict = dict(scores_list)

    # ── Signal 1: Is Task 2 the sole outlier and strategy is noisy? ──────────
    # Task 2 (line localizer) always scores high for heuristic/all_lines strategy
    # because it finds risky structural patterns everywhere in C code.
    if outlier == "task2" and s2 > 0.7 and s1 < 0.3 and s3 < 0.1:
        scores_dict["NOISY_STRATEGY"] += 3
        evidence.append(
            "Task 2 is the sole high-scorer — line localizer inflated by heuristic "
            "pattern matching flagging array/pointer operations in safe code."
        )
        if strategy in ("heuristic", "all_lines"):
            scores_dict["NOISY_STRATEGY"] += 2
            evidence.append(
                f"Strategy='{strategy}' (quality={strat_quality}) is known to produce "
                f"high Task 2 scores even in clean functions."
            )

    # ── Signal 2: Task 3 and Task 4 agree but Task 1 and Task 2 disagree ─────
    # Suggests real constructs/taint but pattern classifier missed it → MISSED_VULN
    if s3 > 0.2 and s4 > 0.3 and s1 < 0.2:
        scores_dict["MISSED_VULN"] += 2
        evidence.append(
            f"Tasks 3 and 4 report risk (T3={s3:.3f}, T4={s4:.3f}) but Task 1 "
            f"pattern classifier shows low score ({s1:.3f}). The vulnerability "
            f"may not match known patterns but has real structural evidence."
        )

    # ── Signal 3: High-severity CWE found by syntax analyser ─────────────────
    if dom_severity in ("HIGH", "MEDIUM") and len(constructs) > 0:
        scores_dict["MISSED_VULN"] += 2
        evidence.append(
            f"Syntax analyser found {len(constructs)} unsafe construct(s) with "
            f"severity={dom_severity} and dominant CWE={vca['dominant_cwe']}. "
            f"These are real code-level risks, not pattern noise."
        )

    # ── Signal 4: Taint propagation paths exist ───────────────────────────────
    if taint_paths > 0:
        scores_dict["MISSED_VULN"] += 1
        evidence.append(
            f"Dependency analyser found {taint_paths} taint propagation path(s) "
            f"from source to sink. Tainted data reaching dangerous sinks "
            f"is a strong indicator of genuine vulnerability."
        )

    # ── Signal 5: Task 1 has pattern hits but other tasks are all low ─────────
    # Many pattern hits without corroboration = possible false pattern match
    t1_hits = 0
    r1_feats = {}
    # We reconstruct hits from the score — if s1 > 0 but s3 == 0 and s4 < 0.2
    if s1 > 0.15 and s3 == 0 and s4 < 0.2 and s2 > 0.7:
        scores_dict["FALSE_PATTERN"] += 2
        evidence.append(
            f"Task 1 reports pattern hits (score={s1:.3f}) but Tasks 3 and 4 "
            f"show no corroborating syntactic or data-flow evidence. "
            f"Pattern hits may be false positives (e.g., safe use of strcpy-like names)."
        )

    # ── Signal 6: Large function with all_lines strategy ─────────────────────
    if strategy == "all_lines" and func_lines > 40 and s2 > 0.6:
        scores_dict["NOISY_STRATEGY"] += 2
        evidence.append(
            f"all_lines strategy on a {func_lines}-line function causes Task 2 "
            f"to flag {int(s2 * func_lines):.0f}+ lines. Most of these are "
            f"structural operations (array access, pointer deref) that are safe."
        )

    # ── Signal 7: V score is near the threshold (0.40 – 0.65) ───────────────
    if 0.35 <= V <= 0.65:
        scores_dict["AMBIGUOUS"] += 3
        evidence.append(
            f"Vulnerability score V={V:.4f} is close to the decision boundary "
            f"(threshold=0.52). The function is genuinely borderline — "
            f"slight changes in any task score would flip the verdict."
        )

    # ── Signal 8: Split disagreement pattern ─────────────────────────────────
    if conflict_pat == "SPLIT_DISAGREEMENT":
        scores_dict["AMBIGUOUS"] += 1
        evidence.append(
            "Tasks split evenly 2 high vs 2 low — no dominant signal direction. "
            "This suggests the function has mixed risk characteristics."
        )

    # ── Determine primary error source ───────────────────────────────────────
    primary_source = max(scores_dict, key=scores_dict.get)
    max_score      = scores_dict[primary_source]

    # If all scores are zero, default to AMBIGUOUS
    if max_score == 0:
        primary_source = "AMBIGUOUS"
        evidence.append(
            "No strong signal for any specific error source. "
            "Function is classified as genuinely ambiguous."
        )

    # Error source descriptions and recommended repair actions
    _descriptions = {
        "MISSED_VULN": (
            "The function likely contains a real vulnerability that one or more "
            "tasks failed to fully detect. Tasks with corroborating evidence "
            "(syntax constructs, taint paths) suggest genuine risk."
        ),
        "FALSE_PATTERN": (
            "One or more tasks are artificially inflated by pattern matches that "
            "do not represent real vulnerabilities. The suspicious patterns may "
            "be safe uses of typically-dangerous APIs, or struct/array accesses "
            "in well-bounded contexts."
        ),
        "NOISY_STRATEGY": (
            "The suspicious-line mapping strategy (heuristic or all_lines) is "
            "producing noisy input that inflates Task 2 scores. The line localizer "
            "flags structural operations (pointer derefs, array accesses) that "
            "are ubiquitous in C code and not inherently dangerous."
        ),
        "AMBIGUOUS": (
            "The function has genuinely mixed risk characteristics. No single "
            "error source dominates. The disagreement reflects real uncertainty "
            "about whether this function is safe or vulnerable."
        ),
    }

    _repair_actions = {
        "MISSED_VULN": [
            "Prioritise for manual code review — focus on lines flagged by Tasks 3 and 4",
            "Examine taint propagation paths from source to sink",
            "Review dangerous constructs identified by syntax analyser",
            "Apply targeted patch for the dominant CWE if confirmed",
        ],
        "FALSE_PATTERN": [
            "Verify pattern hits are genuine by inspecting the flagged lines",
            "Check if dangerous API calls are properly bounds-checked",
            "If safe, mark as reviewed false-positive and exclude from repair",
            "Consider refining Task 1 pattern weights for this code pattern",
        ],
        "NOISY_STRATEGY": [
            "Re-run with ground_truth strategy if CVE-annotated lines are available",
            "Focus diagnosis on lines flagged by Tasks 1, 3, and 4 only",
            "Discount Task 2 score for this sample in the final verdict",
            "Consider this function lower priority for repair",
        ],
        "AMBIGUOUS": [
            "Escalate to human reviewer for final judgement",
            "Apply conservative repair to the highest-risk lines identified",
            "Re-run with a larger context window if possible",
            "Document as uncertain and flag for re-evaluation",
        ],
    }

    log.info(
        f"[Diagnosis/ESD] primary_source={primary_source}  "
        f"confidence_scores={scores_dict}  evidence_count={len(evidence)}"
    )

    return {
        "primary_error_source":      primary_source,
        "error_source_description":  _descriptions[primary_source],
        "confidence_scores":         scores_dict,
        "evidence":                  evidence,
        "recommended_repair_actions": _repair_actions[primary_source],
        "secondary_source": (
            sorted(scores_dict, key=scores_dict.get, reverse=True)[1]
            if len(scores_dict) > 1 else None
        ),
    }


# =============================================================================
# Summary writer
# =============================================================================

def write_summary(diag: dict, out_dir: Path):
    sample_id    = diag["sample_id"]
    dataset      = diag["dataset"]
    mtd_verdict  = diag["mtd_verdict"]
    V            = diag["vulnerability_score"]
    T            = diag["trust_score"]
    alc_decision = diag["alc_decision"]

    rca = diag["stage1_root_cause_analysis"]
    vca = diag["stage2_vulnerability_context"]
    esd = diag["stage3_error_source_detection"]

    lines = [
        "=" * 70,
        f"  DIAGNOSIS REPORT",
        f"  Sample: {sample_id}  |  Dataset: {dataset}",
        f"  MTD: {mtd_verdict}  V={V:.4f}  |  ALC: {alc_decision.upper()}  T={T:.4f}",
        "=" * 70,
        "",
        "── STAGE 1: ROOT CAUSE ANALYSIS ──────────────────────────────────────",
        f"  Outlier task:      {rca['outlier_task']} "
        f"(score={rca['outlier_score']:.4f}, direction={rca['outlier_direction']})",
        f"  Conflict pattern:  {rca['conflict_pattern']}",
        f"  Description:       {rca['pattern_description']}",
        "",
        "  Task Score Breakdown:",
    ]
    for task, expl in rca["task_explanations"].items():
        lines.append(f"    {task}: {expl}")

    lines += [
        "",
        "  Top Conflicting Pairs:",
    ]
    for pair in rca["top_conflict_pairs"]:
        lines.append(
            f"    {pair['task_a']}({pair['score_a']:.3f}) vs "
            f"{pair['task_b']}({pair['score_b']:.3f})  diff={pair['diff']:.3f}"
        )

    lines += [
        "",
        "── STAGE 2: VULNERABILITY CONTEXT ────────────────────────────────────",
        f"  Function:    {vca['function_name']}  ({vca['function_lines']} lines)",
        f"  Strategy:    {vca['strategy']}  (quality={vca['strategy_quality']})",
        f"  Note:        {vca['strategy_note']}",
        f"  Dominant CWE:{vca['dominant_cwe']} ({vca['dominant_severity']})",
        "",
        "  CWE Context:",
    ]
    for cwe, info in vca["cwe_context"].items():
        lines.append(
            f"    {cwe} [{info['severity']}]: {info['description']} "
            f"(detected by: {', '.join(info['detected_by'])})"
        )

    if vca["dangerous_constructs"]:
        lines += ["", "  Dangerous Constructs:"]
        for c in vca["dangerous_constructs"]:
            lines.append(
                f"    Line {c['line']} [{c['severity']}] {c['type']} "
                f"— {c['explanation']}"
            )

    taint = vca["taint_flow"]
    if taint["path_count"] > 0:
        lines += [
            "",
            f"  Taint Flow: {taint['source_count']} sources → "
            f"{taint['sink_count']} sinks via "
            f"{taint['path_count']} path(s)",
            f"  Tainted vars: {taint['tainted_variables']}",
        ]
        for p in taint["propagation_paths"]:
            lines.append(
                f"    {p['var']} → {p['sink']} at line {p['line']}"
            )

    lines += [
        "",
        f"  Complexity: {vca['complexity_note']}",
        "",
        "── STAGE 3: ERROR SOURCE DETECTION ───────────────────────────────────",
        f"  Primary error source: {esd['primary_error_source']}",
        f"  Description: {esd['error_source_description']}",
        "",
        "  Evidence:",
    ]
    for i, ev in enumerate(esd["evidence"], 1):
        lines.append(f"    {i}. {ev}")

    lines += [
        "",
        "  Recommended Repair Actions:",
    ]
    for i, action in enumerate(esd["recommended_repair_actions"], 1):
        lines.append(f"    {i}. {action}")

    lines += [
        "",
        "  Confidence scores per error source:",
    ]
    for src, sc in sorted(esd["confidence_scores"].items(),
                          key=lambda x: x[1], reverse=True):
        lines.append(f"    {src}: {sc}")

    lines += ["", "=" * 70]

    summary_path = out_dir / "diagnosis_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Diagnosis summary → {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnosis Module — runs after ALC flags UNTRUSTWORTHY"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory (same directory ALC wrote alc_result.json to)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run diagnosis even if ALC decision is TRUSTWORTHY"
    )
    out_dir = Path(parser.parse_args().out)

    # ── Read ALC result ───────────────────────────────────────────────────────
    alc_path = out_dir / "alc_result.json"
    if not alc_path.exists():
        log.error(
            f"alc_result.json not found at {alc_path}\n"
            f"Run ALC first:  python alc/run_alc.py --out {out_dir}"
        )
        sys.exit(1)

    alc_result = json.loads(alc_path.read_text(encoding="utf-8"))

    # ── Check if diagnosis is needed ─────────────────────────────────────────
    alc_decision = alc_result.get("decision", "untrustworthy")
    if alc_decision == "trustworthy" and not parser.parse_args().force:
        log.info(
            f"ALC decision is TRUSTWORTHY — Diagnosis not needed.\n"
            f"Use --force to run diagnosis anyway."
        )
        # Write a minimal skipped result so the runner knows we ran
        (out_dir / "diagnosis_result.json").write_text(
            json.dumps({"skipped": True, "reason": "ALC=TRUSTWORTHY"}, indent=2),
            encoding="utf-8"
        )
        sys.exit(0)

    # ── Read MTD result ───────────────────────────────────────────────────────
    mtd_path = out_dir / "mtd_result.json"
    if not mtd_path.exists():
        log.error(f"mtd_result.json not found at {mtd_path}")
        sys.exit(1)

    mtd_result = json.loads(mtd_path.read_text(encoding="utf-8"))

    # ── Extract shared fields ─────────────────────────────────────────────────
    sample_id    = alc_result.get("sample_id",       "?")
    dataset      = alc_result.get("dataset",          "unknown")
    strategy     = alc_result.get("strategy",         "heuristic")
    mtd_verdict  = alc_result.get("mtd_verdict",      "UNKNOWN")
    V            = float(alc_result.get("vulnerability_score", 0.0))
    T            = float(alc_result.get("trust_score",         0.0))
    task_scores  = mtd_result.get("task_scores",  {})
    task_results = mtd_result.get("task_results", {})
    alc_stage1   = alc_result.get("stage1_cross_task_consistency", {})

    log.info(
        f"Diagnosis | id={sample_id}  dataset={dataset}  "
        f"MTD={mtd_verdict}  V={V:.4f}  ALC={alc_decision.upper()}  T={T:.4f}"
    )

    # ── Run three diagnosis stages ────────────────────────────────────────────
    rca = root_cause_analysis(task_scores, task_results, alc_stage1, V)
    vca = vulnerability_context_analysis(task_results, mtd_result, V, strategy)
    esd = error_source_detection(rca, vca, task_scores, V, strategy, mtd_verdict)

    log.info(
        f"Diagnosis complete — "
        f"outlier={rca['outlier_task']}  "
        f"pattern={rca['conflict_pattern']}  "
        f"error_source={esd['primary_error_source']}  "
        f"dominant_cwe={vca['dominant_cwe']}"
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    diag_result = {
        "sample_id":             sample_id,
        "dataset":               dataset,
        "strategy":              strategy,
        "mtd_verdict":           mtd_verdict,
        "vulnerability_score":   V,
        "trust_score":           T,
        "alc_decision":          alc_decision,

        "stage1_root_cause_analysis":       rca,
        "stage2_vulnerability_context":     vca,
        "stage3_error_source_detection":    esd,

        "diagnosis_verdict": {
            "error_source":        esd["primary_error_source"],
            "outlier_task":        rca["outlier_task"],
            "dominant_cwe":        vca["dominant_cwe"],
            "recommended_actions": esd["recommended_repair_actions"],
        },
    }

    (out_dir / "diagnosis_result.json").write_text(
        json.dumps(diag_result, indent=2), encoding="utf-8"
    )
    write_summary(diag_result, out_dir)
    log.info(f"Diagnosis outputs written → {out_dir}")


if __name__ == "__main__":
    main()



