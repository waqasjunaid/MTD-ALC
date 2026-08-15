# # =============================================================================
# # run_bigvul.py
# #
# # Evaluates the full trustworthiness framework (MTD + Diagnosis + Repair)
# # on Big-Vul test set using LineVul as the base vulnerability detector.
# #
# # Workflow:
# # 1. Parse LineVul predictions
# # 2. Run MTD trust detection
# # 3. If untrustworthy → run Diagnosis → run Repair
# #
# # Base detector: LineVul[](https://github.com/awsm-research/LineVul)
# # Dataset: Big-Vul (real-world CVE vulnerabilities)
# # =============================================================================
#
#
# #import csv
# #import json
# #import subprocess
# #import sys
# #from pathlib import Path
#
# #csv.field_size_limit(sys.maxsize)
#
# #ROOT = Path(__file__).resolve().parent
# #OUT = ROOT / "outputs"
# #DIAG_DIR = OUT / "diagnosis"
# #SRC_DIR = OUT / "sources"
#
# #DIAG_DIR.mkdir(exist_ok=True)
# #SRC_DIR.mkdir(exist_ok=True)
#
# #DATA = Path("/mnt/data/junaid/linevul/data/big-vul_dataset/test.csv")
# #RESULTS = ROOT / "bigvul_results.csv"
#
# #MAX_SAMPLES = 10
#
#
# #def run(cmd):
# #    print(f"Running: {' '.join(map(str, cmd))}")
# #    subprocess.run(cmd, check=True)
#
#
# #def get_line_numbers(row):
#  #   nums = []
#
#  #   if "flaw_line_index" in row and row["flaw_line_index"]:
#  #       raw = str(row["flaw_line_index"]).strip().replace("[", "").replace("]", "")
#  #       for p in raw.split(","):
#  #           p = p.strip()
#  #           if not p:
#  #               continue
#  #           try:
#  #               nums.append(int(float(p)) + 1)
# #            except:
# #                pass
#
#  #   return sorted(set([n for n in nums if n > 0]))
#
#
# #print("=== Running Multi-Task Trust Detector + Conditional Diagnosis/Repair on BigVul ===")
#
# #with open(DATA, newline="", encoding="utf-8") as f, open(RESULTS, "w", newline="") as out:
#    # reader = csv.DictReader(f)
#    # writer = csv.writer(out)
#
#   #  writer.writerow(["id", "trust_score", "decision", "ran_diagnosis", "ran_repair"])
#
#  #   processed = 0
#
# #    for row in reader:
#
#  #       line_nums = get_line_numbers(row)
#  #       if not line_nums:
#  #           continue
#
# #        processed += 1
# #        if processed > MAX_SAMPLES:
# #            break
#
#  #       sample_id = row.get("id", f"NEW_SAMPLE_{processed}")
#  #       print(f"\n===== SAMPLE {processed}/{MAX_SAMPLES} → {sample_id} =====\n")
#
# #        func_code = (row.get("func_before") or "").strip()
# #        if not func_code:
# #            continue
#
#         # Save source code
#  #       src_path = SRC_DIR / f"{sample_id}.c"
# #        src_path.write_text(func_code, encoding="utf-8")
#
#   #      sample = {
#   #          "file": str(src_path),
#   #          "suspicious_lines": line_nums
#   #      }
#
#  #       (OUT / "sample_pred.json").write_text(json.dumps(sample, indent=2))
#
#         # Always run MTD (trust detection)
#   #      run([sys.executable, str(ROOT / "mtd/run_mtd.py")])
#
#         # Load MTD decision
#   #      decision_path = OUT / "decision.txt"
#   #      if not decision_path.exists():
#   #          print("MTD failed to produce decision.txt → skipping")
#   #          continue
#
#   #      with open(decision_path, "r") as f:
#   #          decision = f.read().strip()
#
#   #      trust_score = float((OUT / "T_score.txt").read_text())
#
#         # Conditional: only run diagnosis & repair if untrustworthy
#   #      ran_diagnosis = "No"
#   #      ran_repair = "No"
#
#   #      if decision.lower() == "untrustworthy":
#   #          print("→ Untrustworthy → running diagnosis and repair")
#
#             # Run diagnosis
#   #          try:
#   #              run([sys.executable, "-m", "diagnosis.run_diagnosis"])
#   #              diag_file = OUT / "diagnosis_report.json"
#   #              if diag_file.exists():
#    #                 (DIAG_DIR / f"{sample_id}.json").write_text(diag_file.read_text())
#    #             ran_diagnosis = "Yes"
#   #          except Exception as e:
#   #              print("Diagnosis failed:", e)
#
#             # Run repair
#  #           try:
#  #               run([sys.executable, "-m", "repair.run_repair"])
#  #               ran_repair = "Yes"
#  #           except Exception as e:
#  #               print("Repair failed:", e)
#
#  #       else:
#  #           print("→ Trustworthy → skipping diagnosis and repair")
#
# #        writer.writerow([sample_id, round(trust_score, 4), decision, ran_diagnosis, ran_repair])
#
# #print("\n=== DONE — results saved to bigvul_results.csv ===")
# #print(f"Processed {processed} samples")
#
#
#
#
#
# # =============================================================================
# # run_bigvul.py  —  BigVul Full Pipeline
# #
# # Pipeline:
# #   MTD → ALC → Diagnosis (if UNTRUSTWORTHY) → Repair (if UNTRUSTWORTHY)
# #
# # CSV columns added by each stage:
# #   MTD:       task1_score, task2_score, task3_score, task4_score, V_score, mtd_verdict
# #   ALC:       T_score, alc_decision, trust_level
# #   Diagnosis: diagnosis_run, error_source, outlier_task, dominant_cwe, conflict_pattern
# #   Repair:    repair_run, patch_generated, repair_cwe, repair_error_source
# # =============================================================================
#
# import csv
# import json
# import logging
# import subprocess
# import sys
# from pathlib import Path
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# ROOT         = Path(__file__).resolve().parent
# DATA_FILE    = ROOT / "data"    / "bigvul_preprocessed_heuristic.jsonl"
# OUT_DIR      = ROOT / "outputs" / "bigvul"
# RESULTS_CSV  = ROOT / "bigvul_results_heuristic.csv"
# PYTHON       = sys.executable
# MTD_SCRIPT   = ROOT / "mtd"       / "run_mtd.py"
# ALC_SCRIPT   = ROOT / "alc"       / "run_alc.py"
# DIAG_SCRIPT  = ROOT / "diagnosis" / "run_diagnosis.py"
# REPAIR_SCRIPT= ROOT / "repair"    / "run_repair.py"
# MAX_SAMPLES  = 500
#
# CSV_FIELDS = [
#     "sample_id", "dataset", "label", "func_name", "strategy",
#     "total_lines", "suspicious_lines",
#     # MTD
#     "task1_score", "task2_score", "task3_score", "task4_score",
#     "V_score", "mtd_verdict",
#     # ALC
#     "T_score", "alc_decision", "trust_level",
#     # Diagnosis
#     "diagnosis_run", "error_source", "outlier_task",
#     "dominant_cwe", "conflict_pattern",
#     # Repair
#     "repair_run", "patch_generated", "repair_cwe", "repair_error_source",
# ]
#
#
# def run_script(script: Path, out_dir: Path, label: str) -> bool:
#     r = subprocess.run(
#         [PYTHON, str(script), "--out", str(out_dir)],
#         capture_output=False,
#     )
#     if r.returncode != 0:
#         log.error(f"{label} failed (exit {r.returncode})")
#         return False
#     return True
#
#
# def main():
#     log.info("=== BigVul Full Pipeline: MTD → ALC → Diagnosis → Repair ===")
#
#     if not DATA_FILE.exists():
#         log.error(f"Data file not found: {DATA_FILE}")
#         sys.exit(1)
#
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     processed = 0
#
#     with open(DATA_FILE, encoding="utf-8") as in_fh, \
#          open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csv_fh:
#
#         writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS,
#                                 extrasaction="ignore")
#         writer.writeheader()
#
#         for raw_line in in_fh:
#             if processed >= MAX_SAMPLES:
#                 break
#
#             row       = json.loads(raw_line)
#             sample_id = str(row.get("id", processed))
#             label     = int(row.get("label", 0))
#             func_name = row.get("func_name") or row.get("func", {})
#             if isinstance(func_name, dict):
#                 func_name = func_name.get("name", "unknown")
#             strategy         = row.get("line_map", {}).get("strategy", "heuristic")
#             total_lines      = row.get("line_map", {}).get("total_lines", 0)
#             suspicious_lines = len(row.get("suspicious_line_numbers", []))
#
#             log.info(
#                 f"===== SAMPLE {processed+1}/{MAX_SAMPLES}  "
#                 f"id={sample_id}  label={label}  func={func_name} ====="
#             )
#
#             # Write sample_pred.json for MTD
#             (OUT_DIR / "sample_pred.json").write_text(json.dumps({
#                 "sample_id":        sample_id,
#                 "dataset":          "bigvul",
#                 "label":            label,
#                 "file":             row.get("source_file", ""),
#                 "suspicious_lines": row.get("suspicious_line_numbers", []),
#                 "func":             row.get("func", {}),
#                 "line_map":         row.get("line_map", {}),
#             }), encoding="utf-8")
#
#             csv_row = {
#                 "sample_id":          sample_id,
#                 "dataset":            "bigvul",
#                 "label":              label,
#                 "func_name":          func_name,
#                 "strategy":           strategy,
#                 "total_lines":        total_lines,
#                 "suspicious_lines":   suspicious_lines,
#                 "diagnosis_run":      "NO",
#                 "error_source":       "",
#                 "outlier_task":       "",
#                 "dominant_cwe":       "",
#                 "conflict_pattern":   "",
#                 "repair_run":         "NO",
#                 "patch_generated":    "",
#                 "repair_cwe":         "",
#                 "repair_error_source":"",
#             }
#
#             # ── Step 1: MTD ───────────────────────────────────────────────────
#             if not run_script(MTD_SCRIPT, OUT_DIR, "MTD"):
#                 processed += 1; continue
#
#             mtd_path = OUT_DIR / "mtd_result.json"
#             if not mtd_path.exists():
#                 processed += 1; continue
#
#             mtd         = json.loads(mtd_path.read_text(encoding="utf-8"))
#             ts          = mtd.get("task_scores", {})
#             vuln_block  = mtd.get("vulnerability", {})
#             V           = float(vuln_block.get("score", 0.0))
#             mtd_verdict = vuln_block.get("label", "UNKNOWN")
#
#             csv_row.update({
#                 "task1_score": round(ts.get("task1", 0), 4),
#                 "task2_score": round(ts.get("task2", 0), 4),
#                 "task3_score": round(ts.get("task3", 0), 4),
#                 "task4_score": round(ts.get("task4", 0), 4),
#                 "V_score":     round(V, 4),
#                 "mtd_verdict": mtd_verdict,
#             })
#
#             # ── Step 2: ALC ───────────────────────────────────────────────────
#             if not run_script(ALC_SCRIPT, OUT_DIR, "ALC"):
#                 processed += 1; continue
#
#             alc_path = OUT_DIR / "alc_result.json"
#             if not alc_path.exists():
#                 processed += 1; continue
#
#             alc          = json.loads(alc_path.read_text(encoding="utf-8"))
#             T            = float(alc.get("trust_score", 0.0))
#             alc_decision = alc.get("decision", "untrustworthy")
#             trust_level  = alc.get(
#                 "stage3_trust_score_computation", {}
#             ).get("trust_level", "LOW")
#
#             csv_row.update({
#                 "T_score":      round(T, 4),
#                 "alc_decision": alc_decision,
#                 "trust_level":  trust_level,
#             })
#
#             # ── Step 3: Diagnosis (only if UNTRUSTWORTHY) ────────────────────
#             if alc_decision == "untrustworthy" and DIAG_SCRIPT.exists():
#                 log.info(f"ALC=UNTRUSTWORTHY → running Diagnosis id={sample_id}")
#                 run_script(DIAG_SCRIPT, OUT_DIR, "Diagnosis")
#
#                 diag_path = OUT_DIR / "diagnosis_result.json"
#                 if diag_path.exists():
#                     diag = json.loads(diag_path.read_text(encoding="utf-8"))
#                     if not diag.get("skipped"):
#                         esd = diag.get("stage3_error_source_detection", {})
#                         rca = diag.get("stage1_root_cause_analysis",    {})
#                         vca = diag.get("stage2_vulnerability_context",  {})
#                         csv_row.update({
#                             "diagnosis_run":    "YES",
#                             "error_source":     esd.get("primary_error_source", ""),
#                             "outlier_task":     rca.get("outlier_task", ""),
#                             "dominant_cwe":     vca.get("dominant_cwe", "") or "",
#                             "conflict_pattern": rca.get("conflict_pattern", ""),
#                         })
#
#                         # ── Step 4: Repair (after Diagnosis) ─────────────────
#                         if REPAIR_SCRIPT.exists():
#                             log.info(f"Running Repair id={sample_id}")
#                             run_script(REPAIR_SCRIPT, OUT_DIR, "Repair")
#
#                             repair_path = OUT_DIR / "repair_result.json"
#                             if repair_path.exists():
#                                 rpr = json.loads(repair_path.read_text(encoding="utf-8"))
#                                 if not rpr.get("skipped"):
#                                     rv = rpr.get("repair_verdict", {})
#                                     csv_row.update({
#                                         "repair_run":          "YES",
#                                         "patch_generated":     str(rv.get("patch_generated", "")),
#                                         "repair_cwe":          rv.get("dominant_cwe", ""),
#                                         "repair_error_source": rv.get("error_source", ""),
#                                     })
#                                     log.info(
#                                         f"Repair complete — "
#                                         f"patch={rv.get('patch_generated')}  "
#                                         f"cwe={rv.get('dominant_cwe')}"
#                                     )
#             else:
#                 log.info(f"ALC=TRUSTWORTHY → Diagnosis+Repair skipped id={sample_id}")
#
#             writer.writerow(csv_row)
#             csv_fh.flush()
#             processed += 1
#
#     log.info(f"=== DONE — {RESULTS_CSV}  ({processed} samples) ===")
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#

# =============================================================================
# run_bigvul.py  —  BigVul Full Pipeline
#
# Pipeline:
#   MTD → ALC → Diagnosis (if UNTRUSTWORTHY) → Repair (if UNTRUSTWORTHY)
#
# CSV columns added by each stage:
#   MTD:       task1_score, task2_score, task3_score, task4_score, V_score, mtd_verdict
#   ALC:       T_score, alc_decision, trust_level
#   Diagnosis: diagnosis_run, error_source, outlier_task, dominant_cwe, conflict_pattern
#   Repair:    repair_run, patch_generated, repair_cwe, repair_error_source
# =============================================================================

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent
DATA_FILE    = ROOT / "data"    / "bigvul_preprocessed_heuristic.jsonl"
OUT_DIR      = ROOT / "outputs" / "bigvul"
RESULTS_CSV  = ROOT / "bigvul_results_heuristic.csv"
PYTHON       = sys.executable
MTD_SCRIPT   = ROOT / "mtd"       / "run_mtd.py"
ALC_SCRIPT   = ROOT / "alc"       / "run_alc.py"
DIAG_SCRIPT  = ROOT / "diagnosis" / "run_diagnosis.py"
REPAIR_SCRIPT= ROOT / "repair"    / "run_repair.py"
# Set MAX_SAMPLES = None to run on the full dataset
# Set MAX_SAMPLES = N   to limit to N samples (useful for testing)
MAX_SAMPLES  = None

CSV_FIELDS = [
    "sample_id", "dataset", "label", "func_name", "strategy",
    "total_lines", "suspicious_lines",
    # MTD
    "task1_score", "task2_score", "task3_score", "task4_score",
    "V_score", "mtd_verdict",
    # ALC
    "T_score", "alc_decision", "trust_level",
    # Diagnosis
    "diagnosis_run", "error_source", "outlier_task",
    "dominant_cwe", "conflict_pattern",
    # Repair
    "repair_run", "patch_generated", "repair_cwe", "repair_error_source",
]


def run_script(script: Path, out_dir: Path, label: str) -> bool:
    r = subprocess.run(
        [PYTHON, str(script), "--out", str(out_dir)],
        capture_output=False,
    )
    if r.returncode != 0:
        log.error(f"{label} failed (exit {r.returncode})")
        return False
    return True


def main():
    n_str = str(MAX_SAMPLES) if MAX_SAMPLES is not None else "ALL"
    log.info(f"=== BigVul Full Pipeline: MTD → ALC → Diagnosis → Repair  (max={n_str}) ===")

    if not DATA_FILE.exists():
        log.error(f"Data file not found: {DATA_FILE}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0

    with open(DATA_FILE, encoding="utf-8") as in_fh, \
         open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csv_fh:

        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()

        for raw_line in in_fh:
            if MAX_SAMPLES is not None and processed >= MAX_SAMPLES:
                break

            row       = json.loads(raw_line)
            sample_id = str(row.get("id", processed))
            label     = int(row.get("label", 0))
            func_name = row.get("func_name") or row.get("func", {})
            if isinstance(func_name, dict):
                func_name = func_name.get("name", "unknown")
            strategy         = row.get("line_map", {}).get("strategy", "heuristic")
            total_lines      = row.get("line_map", {}).get("total_lines", 0)
            suspicious_lines = len(row.get("suspicious_line_numbers", []))

            total_str = str(MAX_SAMPLES) if MAX_SAMPLES is not None else "?"
            log.info(
                f"===== SAMPLE {processed+1}/{total_str}  "
                f"id={sample_id}  label={label}  func={func_name} ====="
            )

            # Write sample_pred.json for MTD
            (OUT_DIR / "sample_pred.json").write_text(json.dumps({
                "sample_id":        sample_id,
                "dataset":          "bigvul",
                "label":            label,
                "file":             row.get("source_file", ""),
                "suspicious_lines": row.get("suspicious_line_numbers", []),
                "func":             row.get("func", {}),
                "line_map":         row.get("line_map", {}),
            }), encoding="utf-8")

            csv_row = {
                "sample_id":          sample_id,
                "dataset":            "bigvul",
                "label":              label,
                "func_name":          func_name,
                "strategy":           strategy,
                "total_lines":        total_lines,
                "suspicious_lines":   suspicious_lines,
                "diagnosis_run":      "NO",
                "error_source":       "",
                "outlier_task":       "",
                "dominant_cwe":       "",
                "conflict_pattern":   "",
                "repair_run":         "NO",
                "patch_generated":    "",
                "repair_cwe":         "",
                "repair_error_source":"",
            }

            # ── Step 1: MTD ───────────────────────────────────────────────────
            if not run_script(MTD_SCRIPT, OUT_DIR, "MTD"):
                processed += 1; continue

            mtd_path = OUT_DIR / "mtd_result.json"
            if not mtd_path.exists():
                processed += 1; continue

            mtd         = json.loads(mtd_path.read_text(encoding="utf-8"))
            ts          = mtd.get("task_scores", {})
            vuln_block  = mtd.get("vulnerability", {})
            V           = float(vuln_block.get("score", 0.0))
            mtd_verdict = vuln_block.get("label", "UNKNOWN")

            csv_row.update({
                "task1_score": round(ts.get("task1", 0), 4),
                "task2_score": round(ts.get("task2", 0), 4),
                "task3_score": round(ts.get("task3", 0), 4),
                "task4_score": round(ts.get("task4", 0), 4),
                "V_score":     round(V, 4),
                "mtd_verdict": mtd_verdict,
            })

            # ── Step 2: ALC ───────────────────────────────────────────────────
            if not run_script(ALC_SCRIPT, OUT_DIR, "ALC"):
                processed += 1; continue

            alc_path = OUT_DIR / "alc_result.json"
            if not alc_path.exists():
                processed += 1; continue

            alc          = json.loads(alc_path.read_text(encoding="utf-8"))
            T            = float(alc.get("trust_score", 0.0))
            alc_decision = alc.get("decision", "untrustworthy")
            trust_level  = alc.get(
                "stage3_trust_score_computation", {}
            ).get("trust_level", "LOW")

            csv_row.update({
                "T_score":      round(T, 4),
                "alc_decision": alc_decision,
                "trust_level":  trust_level,
            })

            # ── Step 3: Diagnosis (only if UNTRUSTWORTHY) ────────────────────
            if alc_decision == "untrustworthy" and DIAG_SCRIPT.exists():
                log.info(f"ALC=UNTRUSTWORTHY → running Diagnosis id={sample_id}")
                run_script(DIAG_SCRIPT, OUT_DIR, "Diagnosis")

                diag_path = OUT_DIR / "diagnosis_result.json"
                if diag_path.exists():
                    diag = json.loads(diag_path.read_text(encoding="utf-8"))
                    if not diag.get("skipped"):
                        esd = diag.get("stage3_error_source_detection", {})
                        rca = diag.get("stage1_root_cause_analysis",    {})
                        vca = diag.get("stage2_vulnerability_context",  {})
                        csv_row.update({
                            "diagnosis_run":    "YES",
                            "error_source":     esd.get("primary_error_source", ""),
                            "outlier_task":     rca.get("outlier_task", ""),
                            "dominant_cwe":     vca.get("dominant_cwe", "") or "",
                            "conflict_pattern": rca.get("conflict_pattern", ""),
                        })

                        # ── Step 4: Repair (after Diagnosis) ─────────────────
                        if REPAIR_SCRIPT.exists():
                            log.info(f"Running Repair id={sample_id}")
                            run_script(REPAIR_SCRIPT, OUT_DIR, "Repair")

                            repair_path = OUT_DIR / "repair_result.json"
                            if repair_path.exists():
                                rpr = json.loads(repair_path.read_text(encoding="utf-8"))
                                if not rpr.get("skipped"):
                                    rv = rpr.get("repair_verdict", {})
                                    csv_row.update({
                                        "repair_run":          "YES",
                                        "patch_generated":     str(rv.get("patch_generated", "")),
                                        "repair_cwe":          rv.get("dominant_cwe", ""),
                                        "repair_error_source": rv.get("error_source", ""),
                                    })
                                    log.info(
                                        f"Repair complete — "
                                        f"patch={rv.get('patch_generated')}  "
                                        f"cwe={rv.get('dominant_cwe')}"
                                    )
            else:
                log.info(f"ALC=TRUSTWORTHY → Diagnosis+Repair skipped id={sample_id}")

            writer.writerow(csv_row)
            csv_fh.flush()
            processed += 1

    log.info(f"=== DONE — {RESULTS_CSV}  ({processed} samples) ===")


if __name__ == "__main__":
    main()



