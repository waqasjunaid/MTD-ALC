# =============================================================================
# alc/run_alc.py  —  Adaptive Logic Calibration (ALC) Trustworthiness Module
#
# FRAMEWORK POSITION:
#   Preprocessing → MTD → [ALC] → Trustworthy / Untrustworthy
#                                        ↓                ↓
#                                     Accept     Diagnosis → Repair
#
# ALC is SEPARATE from MTD. It reads mtd_result.json, runs three stages,
# and writes the trust decision.
#
# This file is the orchestrator only. The three stage implementations live in:
#   alc/cross_task_consistency.py   Stage 1 — CTC
#   alc/risk_aggregation.py         Stage 2 — RA
#   alc/trust_score_computation.py  Stage 3 — TSC
#
# ZERO HARDCODED CONSTANTS anywhere in the ALC module.
# All numeric values are loaded from the trained model:
#
#   infer.get_trust_threshold()  → T decision boundary
#   infer.get_trust_calibration()→ {intercept, slope}
#   infer.get_task_weights()     → task importance weights
#   infer.get_alc_params()       → all remaining ALC constants:
#       conflict_threshold    75th percentile of val-set pairwise diffs
#       min_consistency       accuracy at max-variance val samples
#       variance_decay        derived: -ln(min_consistency) / max_variance
#       direction_threshold   optimal score boundary from val-set labels
#       blend_weights         {consistency, calibration, strategy} from lstsq
#       strategy_quality      {ground_truth, heuristic, all_lines} pipeline constants
#
# Inputs  (from MTD output directory):
#   mtd_result.json   — task scores, V, all task details
#
# Outputs:
#   alc_result.json   — full three-stage ALC report
#   T_score.txt       — authoritative trust score T
#   decision.txt      — "trustworthy" | "untrustworthy"
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

ALC_DIR = Path(__file__).resolve().parent
ROOT    = ALC_DIR.parent

# Import the three stage modules from the same alc/ directory
sys.path.insert(0, str(ALC_DIR))
import cross_task_consistency  as stage1
import risk_aggregation        as stage2
import trust_score_computation as stage3

# Import ML inference utilities from mtd/ml
sys.path.insert(0, str(ROOT / "mtd" / "ml"))
try:
    from infer import (
        get_trust_threshold, get_trust_calibration,
        get_task_weights, get_alc_params,
        models_available, ModelNotTrainedError,
    )
    _ML_OK = True
except ImportError as e:
    _ML_OK = False
    _ML_ERR = str(e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="ALC Trustworthiness Module — runs after MTD"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory (the same directory MTD wrote mtd_result.json to)"
    )
    out_dir = Path(parser.parse_args().out)

    # ── Check ML availability ─────────────────────────────────────────────────
    if not _ML_OK:
        log.error(
            f"ML import failed: {_ML_ERR}\n"
            f"Install dependencies: pip install numpy"
        )
        sys.exit(1)

    try:
        avail = models_available()
    except Exception as e:
        log.error(f"Cannot check models: {e}")
        sys.exit(1)

    if not avail.get("logreg") and not avail.get("nn"):
        log.error(
            "No trained models found.\n"
            "Run these first:\n"
            "  python mtd/ml/build_dataset.py --dataset all\n"
            "  python mtd/ml/train.py --model both"
        )
        sys.exit(1)

    # ── Load ALL parameters from the trained model — nothing hardcoded ────────
    try:
        trust_threshold = get_trust_threshold()
        trust_cal       = get_trust_calibration()
        task_weights    = get_task_weights()
        alc_params      = get_alc_params()
    except ModelNotTrainedError as e:
        log.error(str(e))
        sys.exit(1)

    log.info(
        f"ALC params loaded from model —\n"
        f"  trust_threshold    = {trust_threshold}\n"
        f"  trust_calibration  = {trust_cal}\n"
        f"  task_weights       = {task_weights}\n"
        f"  conflict_threshold = {alc_params['conflict_threshold']}\n"
        f"  min_consistency    = {alc_params['min_consistency']}\n"
        f"  variance_decay     = {alc_params['variance_decay']}\n"
        f"  direction_threshold= {alc_params['direction_threshold']}\n"
        f"  blend_weights      = {alc_params['blend_weights']}\n"
        f"  strategy_quality   = {alc_params['strategy_quality']}"
    )

    # ── Read MTD result ───────────────────────────────────────────────────────
    mtd_path = out_dir / "mtd_result.json"
    if not mtd_path.exists():
        log.error(
            f"mtd_result.json not found at {mtd_path}\n"
            f"Run MTD first:  python mtd/run_mtd.py --out {out_dir}"
        )
        sys.exit(1)

    mtd_result  = json.loads(mtd_path.read_text(encoding="utf-8"))
    sample_id   = mtd_result.get("sample_id",  "?")
    dataset     = mtd_result.get("dataset",     "unknown")
    label       = mtd_result.get("label",       -1)
    strategy    = mtd_result.get("strategy",    "heuristic")
    task_scores = mtd_result.get("task_scores", {})
    task_results= mtd_result.get("task_results",{})
    line_map    = mtd_result.get("line_map",    {})
    vuln_block  = mtd_result.get("vulnerability", {})
    # MTD writes V under "score" key inside the "vulnerability" block
    V          = float(vuln_block.get("score",
                       vuln_block.get("vulnerability_score",
                       mtd_result.get("fusion", {}).get("vulnerability_score", 0.0))))
    vuln_label = vuln_block.get("label", "UNKNOWN")   # VULNERABLE | NON_VULNERABLE

    log.info(
        f"ALC | id={sample_id}  dataset={dataset}  label={label}  "
        f"strategy={strategy}  "
        f"MTD_verdict={vuln_label}  V={V:.4f}  "
        f"task_scores={task_scores}"
    )

    # ── Stage 1: Cross-Task Consistency (CTC) ────────────────────────────────
    log.info("--- ALC Stage 1: Cross-Task Consistency ---")
    ctc_result = stage1.compute(
        task_scores  = task_scores,
        alc_params   = alc_params,   # all constants come from model
        task_results = task_results,
    )

    # ── Stage 2: Risk Aggregation (RA) ───────────────────────────────────────
    log.info("--- ALC Stage 2: Risk Aggregation ---")
    ra_result = stage2.compute(
        task_scores  = task_scores,
        task_weights = task_weights,  # learned from training
        strategy     = strategy,
        alc_params   = alc_params,    # strategy_quality from model
        line_map     = line_map,
        task_results = task_results,
    )

    # ── Stage 3: Trust Score Computation (TSC) ───────────────────────────────
    log.info("--- ALC Stage 3: Trust Score Computation ---")
    tsc_result = stage3.compute(
        consistency_score = ctc_result["consistency_score"],
        V                 = V,
        strategy          = strategy,
        trust_threshold   = trust_threshold,  # learned from training
        trust_cal         = trust_cal,         # learned from training
        alc_params        = alc_params,        # blend_weights from model
    )

    T        = tsc_result["trust_score"]
    decision = tsc_result["decision"]

    log.info(
        f"ALC complete — "
        f"MTD={vuln_label}  V={V:.4f}  "
        f"T={T:.4f}  level={tsc_result['trust_level']}  "
        f"ALC={decision.upper()}  "
        f"(T={T:.4f} {'<' if T < trust_threshold else '>='} "
        f"trust_threshold={trust_threshold})"
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    alc_result = {
        "sample_id":              sample_id,
        "dataset":                dataset,
        "label":                  label,
        "strategy":               strategy,
        "mtd_verdict":            vuln_label,      # MTD's vulnerability decision
        "vulnerability_score":    V,
        "trust_score":            T,
        "decision":               decision,         # ALC's trustworthiness decision

        "stage1_cross_task_consistency":  ctc_result,
        "stage2_risk_aggregation":        ra_result,
        "stage3_trust_score_computation": tsc_result,

        # Full record of all learned params used — makes every run auditable
        "learned_params_used": {
            "trust_threshold":  trust_threshold,
            "trust_cal":        trust_cal,
            "task_weights":     task_weights,
            "alc_params":       alc_params,
        },
    }

    (out_dir / "alc_result.json").write_text(
        json.dumps(alc_result, indent=2), encoding="utf-8"
    )
    # T_score.txt and decision.txt are authoritative ALC outputs
    # (MTD does not write these — they belong to ALC)
    (out_dir / "T_score.txt").write_text(str(T),    encoding="utf-8")
    (out_dir / "decision.txt").write_text(decision, encoding="utf-8")

    log.info(f"ALC outputs written → {out_dir}")


if __name__ == "__main__":
    main()

