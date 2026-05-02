# =============================================================================
# alc/cross_task_consistency.py  —  ALC Stage 1: Cross-Task Consistency
#
# Position in the framework (from the methodology figure):
#
#   [Preprocessing] → [MTD: Task1/2/3/4] → [ALC Trustworthiness Module]
#                                                     ↓
#                                         Stage 1: Cross-Task Consistency
#                                         Stage 2: Risk Aggregation
#                                         Stage 3: Trust Score Computation
#                                                     ↓
#                                    [Trustworthy] or [Untrustworthy]
#
# ZERO HARDCODED CONSTANTS.
# Every numeric value used here comes from alc_params, which is learned from
# the validation set in train.py (learn_alc_params) and loaded at inference
# time via infer.get_alc_params().
#
# alc_params must contain:
#   conflict_threshold  — pairwise score diff that counts as "conflicting"
#                         learned as 75th percentile of val-set pairwise diffs
#   min_consistency     — floor (lowest achievable consistency score)
#                         learned as prediction accuracy at max-variance samples
#   variance_decay      — steepness of exp(-variance × decay)
#                         derived: -ln(min_consistency) / max_val_variance
#   direction_threshold — score boundary separating "risky" from "clean" votes
#                         found by grid search on val-set labels
#
# What "consistency" means:
#   The four MTD tasks measure different aspects of vulnerability.
#   When all four agree (scores close together) → coherent evidence → high trust.
#   When tasks wildly disagree → incoherent signal → untrustworthy → diagnosis.
#
# Three sub-signals:
#   (a) Variance consistency  exp(-variance × variance_decay)
#   (b) Conflict count        pairs with |s_i - s_j| > conflict_threshold
#   (c) Directional agreement fraction of tasks voting same direction
# =============================================================================

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)


def compute(task_scores:  dict,
            alc_params:   dict,
            task_results: Optional[dict] = None) -> dict:
    """
    Compute cross-task consistency score and full CTC report.

    Parameters
    ----------
    task_scores : dict
        {"task1": float, "task2": float, "task3": float, "task4": float}

    alc_params : dict
        Learned constants from infer.get_alc_params(). Must contain:
          conflict_threshold, min_consistency,
          variance_decay, direction_threshold

    task_results : dict, optional
        Raw task outputs from run_mtd.py for evidence detail.

    Returns
    -------
    dict — full CTC report including consistency_score and params_used
    """
    # Pull all constants from alc_params — nothing hardcoded in this file
    conflict_thr = alc_params["conflict_threshold"]
    min_cons     = alc_params["min_consistency"]
    var_decay    = alc_params["variance_decay"]
    dir_thr      = alc_params["direction_threshold"]

    scores     = list(task_scores.values())
    task_names = list(task_scores.keys())
    n          = len(scores)

    if n == 0:
        return _empty_report(alc_params)

    mean_s   = sum(scores) / n
    variance = sum((s - mean_s) ** 2 for s in scores) / n

    # (a) Variance-based consistency — steepness from learned variance_decay
    var_consistency = math.exp(-variance * var_decay)

    # (b) Pairwise conflicts — threshold from learned conflict_threshold
    pairs = []
    conflict_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff     = abs(scores[i] - scores[j])
            conflict = diff > conflict_thr
            if conflict:
                conflict_count += 1
            pairs.append({
                "task_a":   task_names[i],
                "task_b":   task_names[j],
                "score_a":  round(scores[i], 4),
                "score_b":  round(scores[j], 4),
                "diff":     round(diff, 4),
                "conflict": conflict,
            })

    max_pairs        = n * (n - 1) // 2
    conflict_rate    = conflict_count / max_pairs
    conflict_penalty = conflict_rate * 0.40

    # (c) Directional agreement — boundary from learned direction_threshold
    risky_votes         = sum(1 for s in scores if s >= dir_thr)
    clean_votes         = n - risky_votes
    direction_agreement = abs(risky_votes - clean_votes) / n
    direction_bonus     = direction_agreement * 0.15

    # Outlier task (furthest from mean)
    deviations    = {k: abs(v - mean_s) for k, v in task_scores.items()}
    dominant_task = max(deviations, key=deviations.get)

    # Per-task detail
    task_detail = {
        k: {
            "score":      round(v, 4),
            "deviation":  round(abs(v - mean_s), 4),
            "is_outlier": abs(v - mean_s) > conflict_thr,
            "vote":       "risky" if v >= dir_thr else "clean",
        }
        for k, v in task_scores.items()
    }

    # Final consistency — floor from learned min_consistency
    raw_consistency   = var_consistency - conflict_penalty + direction_bonus
    consistency_score = round(min(1.0, max(min_cons, raw_consistency)), 4)

    evidence_summary = _extract_evidence(task_results) if task_results else {}

    log.info(
        f"[ALC/CTC] consistency={consistency_score:.4f}  "
        f"variance={variance:.4f}  var_decay={var_decay}  "
        f"conflicts={conflict_count}/{max_pairs}  "
        f"conflict_thr={conflict_thr}  "
        f"direction_agreement={direction_agreement:.4f}  "
        f"dir_thr={dir_thr}  dominant={dominant_task}"
    )

    return {
        "consistency_score":   consistency_score,
        "variance":            round(variance,            4),
        "var_consistency":     round(var_consistency,     4),
        "mean_task_score":     round(mean_s,              4),
        "conflict_count":      conflict_count,
        "conflict_rate":       round(conflict_rate,       4),
        "conflict_penalty":    round(conflict_penalty,    4),
        "direction_agreement": round(direction_agreement, 4),
        "direction_bonus":     round(direction_bonus,     4),
        "dominant_task":       dominant_task,
        "pairwise_pairs":      pairs,
        "task_detail":         task_detail,
        "evidence_summary":    evidence_summary,
        # Record the learned values used — makes every run fully auditable
        "params_used": {
            "conflict_threshold":  conflict_thr,
            "min_consistency":     min_cons,
            "variance_decay":      var_decay,
            "direction_threshold": dir_thr,
        },
    }


def _extract_evidence(task_results: dict) -> dict:
    r1 = task_results.get("task1", {})
    r2 = task_results.get("task2", {})
    r3 = task_results.get("task3", {})
    r4 = task_results.get("task4", {})
    return {
        "task1_pattern_hits":  r1.get("features", {}).get("pattern_hit_count", 0),
        "task1_cwe_tags":      r1.get("features", {}).get("cwe_tags", []),
        "task2_risky_lines":   r2.get("summary",  {}).get("risky_line_count",  0),
        "task3_constructs":    len(r3.get("unsafe_constructs", [])),
        "task3_severity":      r3.get("severity_counts", {}),
        "task4_taint_paths":   len(r4.get("data_flow", {}).get("propagation_paths", [])),
        "task4_tainted_vars":  r4.get("data_flow", {}).get("tainted_vars", []),
    }


def _empty_report(alc_params: dict) -> dict:
    return {
        "consistency_score":   alc_params["min_consistency"],
        "variance":            0.0,
        "var_consistency":     1.0,
        "mean_task_score":     0.0,
        "conflict_count":      0,
        "conflict_rate":       0.0,
        "conflict_penalty":    0.0,
        "direction_agreement": 0.0,
        "direction_bonus":     0.0,
        "dominant_task":       "unknown",
        "pairwise_pairs":      [],
        "task_detail":         {},
        "evidence_summary":    {},
        "params_used": {
            "conflict_threshold":  alc_params["conflict_threshold"],
            "min_consistency":     alc_params["min_consistency"],
            "variance_decay":      alc_params["variance_decay"],
            "direction_threshold": alc_params["direction_threshold"],
        },
    }


