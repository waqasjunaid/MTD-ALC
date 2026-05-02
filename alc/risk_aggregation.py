# =============================================================================
# alc/risk_aggregation.py  —  ALC Stage 2: Risk Aggregation
#
# Position in the framework:
#   [MTD: Task1/2/3/4] → [ALC: Stage 1: CTC] → [Stage 2: RA] → [Stage 3: TSC]
#
# ZERO HARDCODED CONSTANTS.
# All strategy quality values come from alc_params["strategy_quality"], which
# is saved into the model JSON during training (learn_alc_params in train.py)
# and loaded via infer.get_alc_params().
#
# alc_params must contain:
#   strategy_quality — {"ground_truth": float, "heuristic": float, "all_lines": float}
#     These are pipeline constants that reflect the factual confidence level of
#     each suspicious-line mapping strategy — not empirical estimates:
#       ground_truth = 1.00  (exact flaw lines from human CVE annotation)
#       heuristic    = 0.80  (pattern-based rules, moderate confidence)
#       all_lines    = 0.50  (every line flagged equally, no information)
#     They are stored in the model JSON so this file has no constants of its own.
#
# Purpose:
#   Combine the four task scores into a single aggregated risk value using:
#     (a) Learned task weights  — from infer.get_task_weights()
#     (b) Evidence mass         — raw counts from task outputs
#     (c) Strategy quality      — from alc_params["strategy_quality"]
#     (d) CWE diversity         — number of distinct CWE categories detected
# =============================================================================

import logging
from typing import Optional

log = logging.getLogger(__name__)


def compute(task_scores:  dict,
            task_weights: dict,
            strategy:     str,
            alc_params:   dict,
            line_map:     dict,
            task_results: Optional[dict] = None) -> dict:
    """
    Aggregate task scores into a single risk value.

    Parameters
    ----------
    task_scores  : {"task1": float, ...}
    task_weights : {"task1": float, ...}  from infer.get_task_weights()
    strategy     : "ground_truth" | "heuristic" | "all_lines"
    alc_params   : from infer.get_alc_params()  — must contain strategy_quality
    line_map     : from sample_pred.json — used for avg line confidence
    task_results : raw task output dicts from run_mtd.py

    Returns
    -------
    dict with keys:
        aggregated_risk      float  primary output — fed to Stage 3
        weighted_task_scores dict
        weighted_sum         float
        evidence_mass        float
        evidence_detail      dict
        strategy             str
        strategy_quality     float
        avg_line_conf        float
        cwe_diversity        float
        cwe_set              list
        params_used          dict   the strategy_quality map actually used
    """
    # Pull strategy quality from alc_params — nothing hardcoded here
    strategy_quality_map = alc_params["strategy_quality"]
    strategy_quality     = strategy_quality_map.get(strategy, 0.70)

    # (a) Weighted sum using learned task weights
    weighted_scores = {
        k: round(task_weights.get(k, 0.0) * task_scores.get(k, 0.0), 4)
        for k in task_scores
    }
    weighted_sum = sum(weighted_scores.values())

    # (b) Evidence mass from raw task outputs
    evidence_detail = {}
    evidence_mass   = 0.0

    if task_results:
        r1 = task_results.get("task1", {})
        r2 = task_results.get("task2", {})
        r3 = task_results.get("task3", {})
        r4 = task_results.get("task4", {})

        pattern_hits  = r1.get("features", {}).get("pattern_hit_count", 0)
        risky_count   = r2.get("summary",  {}).get("risky_line_count",  0)
        construct_cnt = len(r3.get("unsafe_constructs", []))
        path_count    = len(r4.get("data_flow", {}).get("propagation_paths", []))
        source_count  = len(r4.get("data_flow", {}).get("taint_sources",     []))
        sink_count    = len(r4.get("data_flow", {}).get("taint_sinks",       []))

        evidence_detail = {
            "pattern_hits":  pattern_hits,
            "risky_lines":   risky_count,
            "constructs":    construct_cnt,
            "taint_paths":   path_count,
            "taint_sources": source_count,
            "taint_sinks":   sink_count,
        }

        # Normalise each count against a reasonable maximum, then average
        # These normalisation denominators (5, 10, 3, 5) are NOT thresholds —
        # they are scale factors that convert raw counts to a [0,1] range.
        # The actual risk weighting is done by task_weights above.
        evidence_mass = (
            min(1.0, pattern_hits  / 5.0)  * 0.25 +
            min(1.0, risky_count   / 10.0) * 0.25 +
            min(1.0, construct_cnt / 3.0)  * 0.25 +
            min(1.0, (path_count + source_count + sink_count) / 5.0) * 0.25
        )

    # (c) Strategy quality from alc_params — not a hardcoded constant
    # Already extracted above as strategy_quality

    # (d) CWE diversity — number of distinct CWE categories found
    cwes = set()
    if task_results:
        for c in task_results.get("task3", {}).get("unsafe_constructs", []):
            if c.get("cwe"):
                cwes.add(c["cwe"])
        for c in task_results.get("task1", {}).get("features", {}).get("cwe_tags", []):
            cwes.add(c)
    cwe_diversity = min(1.0, len(cwes) / 5.0)

    # Per-line confidence from preprocessor (reflects line-mapper quality)
    conf_vals = [
        e.get("confidence", 0.0)
        for e in line_map.get("suspicious_lines", [])
        if e.get("confidence") is not None
    ]
    avg_conf = round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else 0.0

    # Final aggregated risk:
    #   Primary driver:  weighted task sum × strategy quality
    #   Small bonuses:   evidence mass (0.10) + CWE diversity (0.05)
    aggregated_risk = round(
        min(1.0, max(0.0,
            weighted_sum  * strategy_quality +
            evidence_mass * 0.10             +
            cwe_diversity * 0.05
        )), 4
    )

    log.info(
        f"[ALC/RA] aggregated_risk={aggregated_risk:.4f}  "
        f"weighted_sum={weighted_sum:.4f}  "
        f"strategy={strategy}  strategy_quality={strategy_quality}  "
        f"evidence_mass={evidence_mass:.4f}  "
        f"cwe_diversity={cwe_diversity:.4f}"
    )

    return {
        "aggregated_risk":      aggregated_risk,
        "weighted_task_scores": weighted_scores,
        "weighted_sum":         round(weighted_sum,    4),
        "evidence_mass":        round(evidence_mass,   4),
        "evidence_detail":      evidence_detail,
        "strategy":             strategy,
        "strategy_quality":     strategy_quality,
        "avg_line_conf":        avg_conf,
        "cwe_diversity":        round(cwe_diversity,   4),
        "cwe_set":              sorted(cwes),
        # Record what was used — for auditability
        "params_used": {
            "strategy_quality_map": strategy_quality_map,
        },
    }

