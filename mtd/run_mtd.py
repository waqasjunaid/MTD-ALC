# =============================================================================
# mtd/run_mtd.py  —  Multi-Task Vulnerability Detector (MTD)
#
# Responsibility: run the four tasks and compute the ML vulnerability score V.
# Trust scoring is handled SEPARATELY by alc/run_alc.py (the ALC module).
#
# Framework position:
#   Preprocessing → [MTD] → ALC → Trustworthy/Untrustworthy → Diagnosis/Repair
#
# Outputs:
#   mtd_result.json   — task scores, V, feature vector dim, all task details
#   (T_score.txt and decision.txt are written by ALC, not here)
# =============================================================================

import argparse, json, logging, math, sys
from pathlib import Path

MTD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MTD_DIR))
sys.path.insert(0, str(MTD_DIR / "ml"))

import task1_vulnerability_classification as task1
import task2_line_localization            as task2
import task3_syntax_risk_prediction       as task3
import task4_dependency_propagation_risk  as task4

try:
    from ml.feature_extractor import extract as extract_features
    from ml.infer import (score_ensemble, get_opt_threshold,
                          models_available, ModelNotTrainedError)
    _ML_OK = True
except ImportError as e:
    _ML_OK = False; _ML_ERR = str(e)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MTD — Vulnerability Detector")
    parser.add_argument("--out", required=True)
    out_dir = Path(parser.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _ML_OK:
        log.error(f"ML import failed: {_ML_ERR}\npip install numpy"); sys.exit(1)

    try: avail = models_available()
    except Exception as e: log.error(f"Cannot check models: {e}"); sys.exit(1)

    if not avail.get("logreg") and not avail.get("nn"):
        log.error(
            "No trained models.\n"
            "Run:\n  python mtd/ml/build_dataset.py --dataset all\n"
            "     python mtd/ml/train.py --model both"
        )
        sys.exit(1)

    log.info(f"Models available — logreg={avail['logreg']}  nn={avail['nn']}")

    try:
        opt_threshold = get_opt_threshold()
    except ModelNotTrainedError as e:
        log.error(str(e)); sys.exit(1)

    # ── Read sample_pred.json ─────────────────────────────────────────────────
    pred_path = out_dir / "sample_pred.json"
    if not pred_path.exists():
        log.error(f"sample_pred.json not found at {pred_path}"); sys.exit(1)

    pred             = json.loads(pred_path.read_text(encoding="utf-8"))
    sample_id        = pred.get("sample_id", "?")
    dataset          = pred.get("dataset",   "unknown")
    label            = pred.get("label",     -1)
    source_file      = pred["file"]
    suspicious_lines = pred.get("suspicious_lines", [])
    func             = pred.get("func",     {})
    line_map         = pred.get("line_map", {})
    strategy         = line_map.get("strategy", "heuristic")
    total_lines      = line_map.get("total_lines", 0)

    log.info(
        f"MTD | id={sample_id}  dataset={dataset}  label={label}  "
        f"func={func.get('name','?')}  lines={total_lines}  "
        f"strategy={strategy}  suspicious={len(suspicious_lines)}"
    )

    # ── Run four tasks ────────────────────────────────────────────────────────
    log.info("--- Task 1: Vulnerability Classification ---")
    r1 = task1.run(source_file, suspicious_lines, func, line_map)
    log.info("--- Task 2: Line Localization ---")
    r2 = task2.run(source_file, suspicious_lines, func, line_map)
    log.info("--- Task 3: Syntax Risk Prediction ---")
    r3 = task3.run(source_file, suspicious_lines, func, line_map)
    log.info("--- Task 4: Dependency Propagation Risk ---")
    r4 = task4.run(source_file, suspicious_lines, func, line_map)

    # ── Task scores (scalars for ALC consumption) ─────────────────────────────
    s1 = float(r1.get("score", 0.0))
    s2 = _t2score(r2)
    s3 = float(r3.get("overall_syntax_risk", 0.0))
    s4 = float(r4.get("overall_dependency_risk", 0.0))
    task_scores = {"task1": s1, "task2": s2, "task3": s3, "task4": s4}
    log.info(f"Task scores  T1={s1:.4f}  T2={s2:.4f}  T3={s3:.4f}  T4={s4:.4f}")

    # ── ML vulnerability score V ──────────────────────────────────────────────
    try:
        features = extract_features(r1, r2, r3, r4, func, line_map, suspicious_lines)
        V = round(float(score_ensemble(features)), 4)
    except ModelNotTrainedError as e:
        log.error(str(e)); sys.exit(1)
    except Exception as e:
        log.error(f"ML scoring failed: {e}"); sys.exit(1)

    vuln_label = "VULNERABLE" if V >= opt_threshold else "NON_VULNERABLE"
    log.info(f"Vulnerability score V={V:.4f}  threshold={opt_threshold}  → {vuln_label}")

    # ── Write mtd_result.json — consumed by ALC in next step ─────────────────
    (out_dir / "mtd_result.json").write_text(json.dumps({
        "sample_id":             sample_id,
        "dataset":               dataset,
        "label":                 label,
        "source_file":           source_file,
        "func_name":             func.get("name"),
        "func_lines":            total_lines,
        "strategy":              strategy,
        "suspicious_line_count": len(suspicious_lines),
        "task_scores":           task_scores,
        "task_results":          {"task1":r1,"task2":r2,"task3":r3,"task4":r4},
        # line_map is passed to ALC Stage 2 for per-line confidence scoring
        "line_map":              line_map,
        "vulnerability": {
            "score":          V,           # ALC reads this as "score"
            "label":          vuln_label,  # VULNERABLE | NON_VULNERABLE
            "opt_threshold":  opt_threshold,
            "models_used":    avail,
            "feature_dim":    len(features),
        },
        # NOTE: trust_score and decision are NOT set here.
        # They are written by alc/run_alc.py in the next pipeline step.
    }, indent=2), encoding="utf-8")

    log.info(f"MTD complete — V={V:.4f}  vuln={vuln_label}  "
             f"[ALC step will compute trust score and decision]")


def _t2score(r: dict) -> float:
    s = r.get("summary", {}); total = s.get("total_lines",1) or 1
    risky = s.get("risky_line_count", 0)
    return round(min(1.0, 1.0 - math.exp(-(risky/total)*8.0)), 4)


if __name__ == "__main__":
    main()



