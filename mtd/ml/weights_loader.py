# =============================================================================
# mtd/ml/weights_loader.py
#
# Weights Loader — loads learned pattern weights from trained model files
# and makes them available to all four task modules.
#
# What this solves:
#   The four task files (task1..task4) contain patterns with base weights
#   like 0.90 for strcpy, 0.60 for malloc, etc.  These were initial estimates.
#   After training, the ML model has learned which features actually predict
#   vulnerability.  This module reads those learned importances and exposes
#   rescaled weights that the task files use instead of their hardcoded values.
#
# How rescaling works:
#   1. Load feature_importances from logreg_model.json (LR weights are
#      interpretable; NN weights are not directly interpretable per-feature)
#   2. The feature "pattern_score" (Block A, index 0) aggregates all pattern
#      hits.  Its importance tells us how much patterns matter overall.
#   3. Within patterns, we use the relative importance of CWE-grouped features
#      (unique_cwe_count, high_severity_count, etc.) to redistribute weight
#      across individual patterns.
#   4. For taint sources/sinks, we use data_flow_risk, source_count,
#      sink_count importances to scale individual source/sink severities.
#   5. The rescaled weights are cached at module load time — one disk read
#      per process, then reused for all samples.
#
# If no trained model exists, original base weights are returned unchanged.
# =============================================================================

import json
import logging
import math
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Feature name → index mapping (must match feature_extractor.py FEATURE_NAMES)
_FEAT_IDX = {
    "pattern_score":        0,
    "line_density":         1,
    "avg_line_conf":        2,
    "param_risk":           3,
    "length_risk":          4,
    "risky_line_ratio":     5,
    "top_line_score":       6,
    "mean_risky_score":     7,
    "risky_line_count_log": 8,
    "overall_syntax_risk":  9,
    "high_severity_count":  10,
    "medium_severity_count":11,
    "low_severity_count":   12,
    "unique_cwe_count":     13,
    "construct_density":    14,
    "data_flow_risk":       15,
    "control_flow_risk":    16,
    "overall_dep_risk":     17,
    "tainted_var_count":    18,
    "source_count":         19,
    "sink_count":           20,
    "path_count":           21,
    "nesting_depth_norm":   22,
    "susp_line_ratio":      25,
    "max_line_conf":        26,
    "conf_variance":        28,
}


class _WeightsCache:
    """Singleton that loads learned weights once and caches them."""

    def __init__(self):
        self._loaded      = False
        self._importances = {}     # feature_name → importance float
        self._task_weights = {}    # task1..4 → float
        self._opt_threshold = None

    def _load(self):
        if self._loaded:
            return
        path = MODELS_DIR / "logreg_model.json"
        if not path.exists():
            log.debug("weights_loader: logreg_model.json not found — using base weights")
            self._loaded = True
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("feature_importances", []):
                self._importances[entry["feature"]] = float(entry["importance"])
            self._task_weights   = data.get("task_weights",   {})
            self._opt_threshold  = data.get("opt_threshold",  None)
            log.info(
                f"weights_loader: loaded importances for "
                f"{len(self._importances)} features from {path}"
            )
        except Exception as e:
            log.warning(f"weights_loader: could not load model — {e}")
        self._loaded = True

    def importance(self, feature_name: str, default: float = 0.0) -> float:
        self._load()
        return self._importances.get(feature_name, default)

    def task_weights(self) -> dict:
        self._load()
        return self._task_weights

    def opt_threshold(self) -> Optional[float]:
        self._load()
        return self._opt_threshold

    def rescale_pattern_weight(self, base_weight: float) -> float:
        """
        Rescale a pattern base weight using the learned importance of
        pattern_score and related features.

        Rescaling formula:
          importance_factor = (pattern_score_imp + susp_line_ratio_imp) / 2
          scaled = base_weight * (1 + importance_factor * amplification)

        If no model is loaded, returns base_weight unchanged.
        """
        self._load()
        if not self._importances:
            return base_weight

        # Use pattern_score and susp_line_ratio importances as the signal
        # strength indicator for how much patterns matter
        ps_imp  = self._importances.get("pattern_score",    0.0)
        sl_imp  = self._importances.get("susp_line_ratio",  0.0)
        ld_imp  = self._importances.get("line_density",     0.0)
        avg_imp = (ps_imp + sl_imp + ld_imp) / 3.0

        # Amplification: if patterns are very important (avg_imp > 0.1),
        # boost them; if less important, reduce them slightly
        # Scale factor is centred at 1.0 when avg_imp ≈ 0.05 (neutral)
        amplification = 3.0
        scale = 1.0 + (avg_imp - 0.05) * amplification
        scale = max(0.5, min(2.0, scale))   # clamp to [0.5, 2.0]

        return round(min(1.0, base_weight * scale), 4)

    def rescale_severity_weight(self, base_weight: float,
                                severity: str) -> float:
        """
        Rescale a construct severity weight using learned HIGH/MEDIUM/LOW
        feature importances.
        """
        self._load()
        if not self._importances:
            return base_weight

        key_map = {
            "HIGH":   "high_severity_count",
            "MEDIUM": "medium_severity_count",
            "LOW":    "low_severity_count",
        }
        feat = key_map.get(severity, "overall_syntax_risk")
        sev_imp  = self._importances.get(feat, 0.0)
        all_imp  = self._importances.get("overall_syntax_risk", 0.0)
        avg_imp  = (sev_imp + all_imp) / 2.0

        scale = 1.0 + (avg_imp - 0.03) * 4.0
        scale = max(0.5, min(2.0, scale))

        return round(min(1.0, base_weight * scale), 4)

    def rescale_taint_weight(self, base_weight: float,
                             kind: str = "source") -> float:
        """
        Rescale a taint source or sink severity using learned data_flow
        and source/sink count importances.
        """
        self._load()
        if not self._importances:
            return base_weight

        df_imp  = self._importances.get("data_flow_risk",  0.0)
        src_imp = self._importances.get("source_count",    0.0)
        snk_imp = self._importances.get("sink_count",      0.0)

        if kind == "source":
            avg_imp = (df_imp + src_imp) / 2.0
        else:
            avg_imp = (df_imp + snk_imp) / 2.0

        scale = 1.0 + (avg_imp - 0.02) * 5.0
        scale = max(0.5, min(2.0, scale))

        return round(min(1.0, base_weight * scale), 4)


# Module-level singleton — loaded once, reused across all samples
_cache = _WeightsCache()


def rescale_pattern(base_weight: float) -> float:
    """Rescale a CWE pattern base weight using learned importances."""
    return _cache.rescale_pattern_weight(base_weight)


def rescale_severity(base_weight: float, severity: str) -> float:
    """Rescale a construct severity weight (HIGH/MEDIUM/LOW)."""
    return _cache.rescale_severity_weight(base_weight, severity)


def rescale_taint(base_weight: float, kind: str = "source") -> float:
    """Rescale a taint source/sink severity weight."""
    return _cache.rescale_taint_weight(base_weight, kind)


def get_task_weights() -> dict:
    """Return learned task weights {task1..4}."""
    return _cache.task_weights()


def get_opt_threshold() -> Optional[float]:
    """Return the optimal classification threshold from training."""
    return _cache.opt_threshold()
