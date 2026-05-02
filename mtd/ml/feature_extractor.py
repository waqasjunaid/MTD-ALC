# =============================================================================
# mtd/ml/feature_extractor.py
#
# Extracts a 38-dimensional feature vector from all four task outputs.
# This is the ONLY place features are defined — task files compute raw
# signals and hand them here; this file converts them to a fixed vector.
# =============================================================================

import math
import logging

log = logging.getLogger(__name__)

FEATURE_DIM = 38
FEATURE_NAMES = [
    # Block A — Task 1 (5)
    "pattern_hit_count", "line_density", "avg_line_conf",
    "param_count_norm", "line_count_norm",
    # Block B — Task 2 (5)
    "risky_line_ratio", "top_line_score", "mean_risky_score",
    "risky_line_count_log", "risky_line_count_abs",
    # Block C — Task 3 (6)
    "overall_syntax_risk", "high_severity_count", "medium_severity_count",
    "low_severity_count", "unique_cwe_count", "construct_density",
    # Block D — Task 4 (8)
    "data_flow_risk", "control_flow_risk", "overall_dep_risk",
    "tainted_var_count", "source_count", "sink_count",
    "path_count", "nesting_depth_norm",
    # Block E — metadata (7)
    "susp_line_ratio", "max_line_conf", "min_line_conf",
    "conf_variance", "strategy_gt", "strategy_heuristic", "loop_count_norm",
    # Block F — cross-task interactions (7)
    "t1_x_t3", "t1_x_t4", "t3_x_t4",
    "density_x_conf", "source_x_sink_norm",
    "risky_x_paths", "high_sev_x_data_flow",
]

assert len(FEATURE_NAMES) == FEATURE_DIM


def extract(r1: dict, r2: dict, r3: dict, r4: dict,
            func: dict, line_map: dict, suspicious_lines: list) -> list:
    """Build 38-feature vector from all four task outputs."""

    total_lines = max(1, func.get("line_count") or
                      line_map.get("total_lines") or 1)
    param_count = func.get("param_count", 0) or 0
    strategy    = line_map.get("strategy", "heuristic")

    conf_vals = [e.get("confidence", 0.0)
                 for e in line_map.get("suspicious_lines", [])
                 if e.get("confidence") is not None]

    # ── Block A: Task 1 ──────────────────────────────────────────────────────
    f1      = r1.get("features", {})
    a0 = _s(f1.get("pattern_hit_count", 0) / max(1, total_lines))  # density of hits
    a1 = _s(len(suspicious_lines) / total_lines)                    # line_density
    a2 = _s(sum(conf_vals) / len(conf_vals) if conf_vals else 0.0)  # avg_conf
    a3 = _s(param_count / 10.0)
    a4 = _s(total_lines / 200.0)

    # ── Block B: Task 2 ──────────────────────────────────────────────────────
    f2s         = r2.get("summary", {})
    risky_lines = r2.get("risky_lines", [])
    risky_count = f2s.get("risky_line_count", 0)
    risky_scores= [l.get("risk_score", 0.0) for l in risky_lines]

    b0 = _s(risky_count / total_lines)
    b1 = _s(max(risky_scores) if risky_scores else 0.0)
    b2 = _s(sum(risky_scores) / len(risky_scores) if risky_scores else 0.0)
    b3 = _s(math.log(risky_count + 1))
    b4 = _s(risky_count / 50.0)  # absolute count normalised

    # ── Block C: Task 3 ──────────────────────────────────────────────────────
    constructs  = r3.get("unsafe_constructs", [])
    from collections import Counter
    cwe_tags    = set(c.get("cwe", "") for c in constructs)
    sev_counts  = Counter(c.get("severity") for c in constructs)
    c0 = _s(r3.get("overall_syntax_risk", 0.0))
    c1 = _s(sev_counts.get("HIGH",   0) / 5.0)
    c2 = _s(sev_counts.get("MEDIUM", 0) / 5.0)
    c3 = _s(sev_counts.get("LOW",    0) / 5.0)
    c4 = _s(len(cwe_tags) / 10.0)
    c5 = _s(len(constructs) / max(1, total_lines))

    # ── Block D: Task 4 ──────────────────────────────────────────────────────
    df  = r4.get("data_flow", {})
    cf  = r4.get("control_flow", {})
    d0  = _s(df.get("data_flow_risk",    0.0))
    d1  = _s(cf.get("control_flow_risk", 0.0))
    d2  = _s(r4.get("overall_dependency_risk", 0.0))
    d3  = _s(len(df.get("tainted_vars",      [])) / 10.0)
    d4  = _s(len(df.get("taint_sources",     [])) / 10.0)
    d5  = _s(len(df.get("taint_sinks",       [])) / 10.0)
    d6  = _s(len(df.get("propagation_paths", [])) / 10.0)
    d7  = _s(cf.get("nesting_depth", 0) / 10.0)

    # ── Block E: metadata ────────────────────────────────────────────────────
    e0 = _s(len(suspicious_lines) / total_lines)
    e1 = _s(max(conf_vals) if conf_vals else 0.0)
    e2 = _s(min(conf_vals) if conf_vals else 0.0)
    if len(conf_vals) > 1:
        m = sum(conf_vals) / len(conf_vals)
        e3 = _s(sum((v - m) ** 2 for v in conf_vals) / len(conf_vals))
    else:
        e3 = 0.0
    e4 = 1.0 if strategy == "ground_truth" else 0.0
    e5 = 1.0 if strategy == "heuristic"    else 0.0
    e6 = _s(cf.get("loop_count", 0) / 10.0)

    # ── Block F: interactions ────────────────────────────────────────────────
    t1s = _s(r1.get("score", 0.0))
    f0  = _s(t1s * c0)
    f1_ = _s(t1s * d2)
    f2_ = _s(c0  * d2)
    f3  = _s(a1  * a2)
    f4  = _s(d4  * d5)
    f5  = _s(b0  * d6)
    f6  = _s(c1  * d0)   # high severity × data flow risk

    vector = [
        a0, a1, a2, a3, a4,
        b0, b1, b2, b3, b4,
        c0, c1, c2, c3, c4, c5,
        d0, d1, d2, d3, d4, d5, d6, d7,
        e0, e1, e2, e3, e4, e5, e6,
        f0, f1_, f2_, f3, f4, f5, f6,
    ]
    assert len(vector) == FEATURE_DIM
    return [round(float(v), 6) for v in vector]


def _s(v) -> float:
    """Clamp to [0,1], replace NaN/Inf with 0."""
    try:
        f = float(v)
        return max(0.0, min(1.0, f)) if math.isfinite(f) else 0.0
    except Exception:
        return 0.0


