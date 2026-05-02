# =============================================================================
# alc/trust_score_computation.py  —  ALC Stage 3: Trust Score Computation
#
# Position in the framework:
#   [Stage 1: CTC] + [Stage 2: RA] → [Stage 3: TSC] → decision
#
# ZERO HARDCODED CONSTANTS.
# All blend weights and strategy quality values come from alc_params (model).
# trust_threshold and trust_cal come from infer.get_trust_threshold/calibration.
#
# T blends three components using learned blend_weights:
#   Component 1 (w_consistency)  — consistency_score from Stage 1 (CTC)
#   Component 2 (w_calibration)  — model calibration (decisiveness of V)
#   Component 3 (w_strategy)     — strategy quality of suspicious-line input
#
# Trust levels are defined relative to trust_threshold (not a hardcoded 0.75):
#   HIGH     T >= trust_threshold + 0.10   clearly trustworthy
#   MODERATE trust_threshold <= T          trustworthy but borderline
#   LOW      T < trust_threshold           untrustworthy → diagnosis
#
# Decision:
#   T >= trust_threshold → "trustworthy"
#   T <  trust_threshold → "untrustworthy" → triggers Diagnosis + Repair
#
# NOTE: trust_level and decision are now CONSISTENT by design.
#   LOW  → always untrustworthy
#   MODERATE/HIGH → always trustworthy
#   The old contradiction (HIGH trust but untrustworthy) is eliminated.
# =============================================================================

import logging

log = logging.getLogger(__name__)


def compute(consistency_score: float,
            V:                 float,
            strategy:          str,
            trust_threshold:   float,
            trust_cal:         dict,
            alc_params:        dict) -> dict:
    """
    Compute the final trust score T and trustworthy/untrustworthy decision.

    Parameters
    ----------
    consistency_score : float  — from Stage 1 (CTC)
    V                 : float  — ML vulnerability score from MTD
    strategy          : str    — "ground_truth" | "heuristic" | "all_lines"
    trust_threshold   : float  — learned decision boundary
    trust_cal         : dict   — {"intercept": float, "slope": float}
    alc_params        : dict   — blend_weights + strategy_quality from model

    Returns
    -------
    dict — trust_score, decision, trust_level, components, breakdown
    """
    bw                   = alc_params["blend_weights"]
    w_consistency        = bw["consistency"]
    w_calibration        = bw["calibration"]
    w_strategy           = bw["strategy"]
    strategy_quality_map = alc_params["strategy_quality"]

    # Component 1: cross-task consistency (primary trust signal)
    consistency_component = float(consistency_score)

    # Component 2: model calibration — how decisive is the ML model about V?
    # decisiveness = 0 when V≈0.5 (uncertain), 1 when V≈0 or V≈1 (confident)
    decisiveness          = abs(V - 0.5) * 2.0
    cal_raw               = (trust_cal["intercept"] +
                              trust_cal["slope"] * decisiveness)
    calibration_component = min(1.0, max(0.0, cal_raw))

    # Component 3: strategy quality (from model — not hardcoded)
    strategy_component = strategy_quality_map.get(strategy, 0.70)

    # Weighted blend
    T = (
        w_consistency * consistency_component +
        w_calibration * calibration_component +
        w_strategy    * strategy_component
    )
    T = round(min(1.0, max(0.0, T)), 4)

    # Decision based on learned threshold
    decision = "untrustworthy" if T < trust_threshold else "trustworthy"

    # Trust level — defined RELATIVE to trust_threshold so it is always
    # consistent with the decision (no more HIGH-but-UNTRUSTWORTHY)
    high_boundary     = round(trust_threshold + 0.10, 4)
    if T >= high_boundary:
        trust_level = "HIGH"        # clearly trustworthy
    elif T >= trust_threshold:
        trust_level = "MODERATE"    # trustworthy but close to boundary
    else:
        trust_level = "LOW"         # untrustworthy → triggers diagnosis

    breakdown = {
        "consistency_contribution": round(w_consistency * consistency_component, 4),
        "calibration_contribution": round(w_calibration * calibration_component, 4),
        "strategy_contribution":    round(w_strategy    * strategy_component,    4),
    }

    log.info(
        f"[ALC/TSC] T={T:.4f}  level={trust_level}  decision={decision}  "
        f"consistency={consistency_component:.4f}(×{w_consistency})  "
        f"calibration={calibration_component:.4f}(×{w_calibration})  "
        f"strategy={strategy_component:.2f}(×{w_strategy})  "
        f"threshold={trust_threshold}  "
        f"V={V:.4f}  decisiveness={decisiveness:.4f}"
    )

    return {
        "trust_score":              T,
        "decision":                 decision,
        "trust_level":              trust_level,
        "trust_threshold":          trust_threshold,
        "consistency_component":    round(consistency_component, 4),
        "calibration_component":    round(calibration_component, 4),
        "strategy_component":       round(strategy_component,    4),
        "decisiveness":             round(decisiveness,          4),
        "blend_weights":            bw,
        "contribution_breakdown":   breakdown,
        "trust_calibration_params": trust_cal,
        "params_used": {
            "blend_weights":       bw,
            "strategy_quality":    strategy_quality_map,
            "trust_threshold":     trust_threshold,
            "trust_calibration":   trust_cal,
            "high_boundary":       high_boundary,
        },
    }


