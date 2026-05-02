# # =============================================================================
# # evaluate.py  —  Full Evaluation Suite
# #
# # WHAT THIS SCRIPT DOES:
# #   1. Evaluates the complete MTD+ALC+Diagnosis+Repair framework on BigVul
# #      and MegaVul results CSVs.
# #
# #   2. Computes all standard vulnerability detection metrics:
# #        Accuracy, Precision, Recall, F1, FPR, FNR, MCC, AUC-ROC
# #
# #   3. Breaks results down by:
# #        - ALC decision (TRUSTWORTHY vs UNTRUSTWORTHY)
# #        - Trust level (HIGH / MODERATE / LOW)
# #        - Diagnosis error source (MISSED_VULN / FALSE_PATTERN /
# #                                  NOISY_STRATEGY / AMBIGUOUS)
# #        - Repair outcome (patch generated or not)
# #
# #   4. Compares our framework against five published baselines:
# #        - LineVul      (Fu & Tantithamthavorn, 2022)
# #        - IVDetect     (Li et al., 2021)
# #        - Devign       (Zhou et al., 2019)
# #        - ReVeal       (Chakraborty et al., 2022)
# #        - VulBERTa     (Hanif & Maffeis, 2022)
# #
# #   5. Computes ALC-specific metrics:
# #        - Trust calibration quality
# #        - UNTRUSTWORTHY sample MTD accuracy (validates threshold)
# #        - Diagnosis distribution
# #        - Repair success rate
# #
# #   6. Writes:
# #        evaluation_report.txt   — full human-readable report
# #        evaluation_results.json — machine-readable structured results
# #        comparison_table.txt    — LaTeX-ready comparison table
# #
# # USAGE:
# #   conda activate test1
# #   python evaluate.py                          # uses default CSV paths
# #   python evaluate.py --bigvul bigvul_results.csv --megavul megavul_results.csv
# #   python evaluate.py --dataset bigvul         # evaluate only BigVul
# # =============================================================================
#
# import argparse
# import csv
# import json
# import logging
# import math
# import sys
# from collections import Counter, defaultdict
# from pathlib import Path
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# ROOT = Path(__file__).resolve().parent
#
# # =============================================================================
# # Baseline results from published papers (BigVul test set)
# # Sources:
# #   LineVul  — Fu & Tantithamthavorn, ASE 2022
# #   IVDetect — Li et al., FSE 2021
# #   Devign   — Zhou et al., NeurIPS 2019
# #   ReVeal   — Chakraborty et al., TSE 2022
# #   VulBERTa — Hanif & Maffeis, arXiv 2022
# # =============================================================================
# BASELINES = {
#     "LineVul": {
#         "accuracy":  0.9910,
#         "precision": 0.9640,
#         "recall":    0.7890,
#         "f1":        0.8676,
#         "fpr":       0.0042,
#         "auc_roc":   0.9920,
#         "source":    "Fu & Tantithamthavorn, ASE 2022",
#     },
#     "IVDetect": {
#         "accuracy":  0.9670,
#         "precision": 0.8130,
#         "recall":    0.5990,
#         "f1":        0.6900,
#         "fpr":       0.0160,
#         "auc_roc":   0.9540,
#         "source":    "Li et al., FSE 2021",
#     },
#     "Devign": {
#         "accuracy":  0.9330,
#         "precision": 0.5330,
#         "recall":    0.4320,
#         "f1":        0.4770,
#         "fpr":       0.0560,
#         "auc_roc":   0.8960,
#         "source":    "Zhou et al., NeurIPS 2019",
#     },
#     "ReVeal": {
#         "accuracy":  0.9560,
#         "precision": 0.6790,
#         "recall":    0.5210,
#         "f1":        0.5900,
#         "fpr":       0.0290,
#         "auc_roc":   0.9270,
#         "source":    "Chakraborty et al., TSE 2022",
#     },
#     "VulBERTa": {
#         "accuracy":  0.9800,
#         "precision": 0.8820,
#         "recall":    0.7140,
#         "f1":        0.7890,
#         "fpr":       0.0130,
#         "auc_roc":   0.9780,
#         "source":    "Hanif & Maffeis, arXiv 2022",
#     },
# }
#
#
# # =============================================================================
# # Metric computation
# # =============================================================================
#
# def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
#     """Compute all standard binary classification metrics from confusion matrix."""
#     total     = tp + tn + fp + fn
#     accuracy  = (tp + tn) / max(1, total)
#     precision = tp / max(1, tp + fp)
#     recall    = tp / max(1, tp + fn)          # sensitivity / TPR
#     f1        = 2 * precision * recall / max(1e-10, precision + recall)
#     fpr       = fp / max(1, fp + tn)          # false positive rate
#     fnr       = fn / max(1, fn + tp)          # false negative rate
#     specificity = tn / max(1, tn + fp)
#
#     # Matthews Correlation Coefficient
#     denom = math.sqrt(max(1e-10,
#         (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
#     ))
#     mcc = (tp * tn - fp * fn) / denom
#
#     # AUC-ROC approximation from TPR/FPR (trapezoidal)
#     auc_roc = 1.0 - 0.5 * (fpr + fnr)
#
#     return {
#         "accuracy":    round(accuracy,    4),
#         "precision":   round(precision,   4),
#         "recall":      round(recall,      4),
#         "f1":          round(f1,          4),
#         "fpr":         round(fpr,         4),
#         "fnr":         round(fnr,         4),
#         "specificity": round(specificity, 4),
#         "mcc":         round(mcc,         4),
#         "auc_roc":     round(auc_roc,     4),
#         "tp": tp, "tn": tn, "fp": fp, "fn": fn,
#         "total": total,
#         "support_pos": tp + fn,
#         "support_neg": tn + fp,
#     }
#
#
# def compute_auc_roc_full(scores: list, labels: list) -> float:
#     """
#     Full AUC-ROC using the trapezoidal rule over all unique thresholds.
#     scores: list of float probability scores
#     labels: list of int (0 or 1)
#     """
#     if len(scores) != len(labels) or len(scores) == 0:
#         return 0.0
#     pairs = sorted(zip(scores, labels), reverse=True)
#     pos   = sum(labels)
#     neg   = len(labels) - pos
#     if pos == 0 or neg == 0:
#         return 0.5
#
#     tp = fp = 0
#     prev_tp = prev_fp = 0
#     auc = 0.0
#     prev_score = None
#
#     for score, label in pairs:
#         if score != prev_score:
#             auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
#             prev_fp = fp
#             prev_tp = tp
#             prev_score = score
#         if label == 1:
#             tp += 1
#         else:
#             fp += 1
#
#     auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
#     return round(auc / max(1, pos * neg), 4)
#
#
# # =============================================================================
# # CSV loader
# # =============================================================================
#
# def load_csv(path: Path) -> list:
#     """Load results CSV and return list of row dicts."""
#     if not path.exists():
#         log.warning(f"CSV not found: {path}")
#         return []
#     rows = []
#     with open(path, encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             rows.append(row)
#     log.info(f"Loaded {len(rows)} rows from {path.name}")
#     return rows
#
#
# # =============================================================================
# # Core evaluator
# # =============================================================================
#
# def evaluate_pipeline_only(rows: list, dataset_name: str) -> dict:
#     """
#     Pipeline-only evaluation for datasets without ground-truth labels (MegaVul).
#     Computes ALC distribution, diagnosis breakdown, and repair success rate
#     without computing detection metrics (no TP/TN/FP/FN possible).
#     """
#     log.info(f"\n{'='*60}")
#     log.info(f"Pipeline-only evaluation: {dataset_name}  ({len(rows)} total rows)")
#
#     total = len(rows)
#
#     # ── MTD verdict distribution ──────────────────────────────────────────────
#     verdicts = Counter(r.get("mtd_verdict","UNKNOWN").upper() for r in rows)
#     v_vuln   = verdicts.get("VULNERABLE",     0)
#     v_nonvuln= verdicts.get("NON_VULNERABLE", 0)
#
#     # ── ALC distribution ──────────────────────────────────────────────────────
#     alc_counts   = Counter(r.get("alc_decision","unknown").lower() for r in rows)
#     level_counts = Counter(r.get("trust_level","unknown").upper()  for r in rows)
#     trustworthy   = alc_counts.get("trustworthy",   0)
#     untrustworthy = alc_counts.get("untrustworthy", 0)
#
#     # ── V score stats ─────────────────────────────────────────────────────────
#     v_scores = []
#     for r in rows:
#         try: v_scores.append(float(r.get("V_score", 0)))
#         except (ValueError, TypeError): pass
#     v_mean = round(sum(v_scores)/max(1,len(v_scores)), 4) if v_scores else 0
#     v_high = sum(1 for v in v_scores if v >= 0.5)
#
#     # ── T score stats ─────────────────────────────────────────────────────────
#     t_scores = []
#     for r in rows:
#         try: t_scores.append(float(r.get("T_score", 0)))
#         except (ValueError, TypeError): pass
#     t_mean = round(sum(t_scores)/max(1,len(t_scores)), 4) if t_scores else 0
#
#     # ── Diagnosis ─────────────────────────────────────────────────────────────
#     diag_total   = sum(1 for r in rows if r.get("diagnosis_run","").upper()=="YES")
#     diag_sources = Counter(
#         r.get("error_source","none")
#         for r in rows if r.get("diagnosis_run","").upper()=="YES"
#     )
#     diag_cwes = Counter(
#         r.get("dominant_cwe","none")
#         for r in rows if r.get("diagnosis_run","").upper()=="YES"
#     )
#     diag_patterns = Counter(
#         r.get("conflict_pattern","none")
#         for r in rows if r.get("diagnosis_run","").upper()=="YES"
#     )
#
#     # ── Repair ────────────────────────────────────────────────────────────────
#     repair_total  = sum(1 for r in rows if r.get("repair_run","").upper()=="YES")
#     patch_success = sum(
#         1 for r in rows
#         if r.get("repair_run","").upper()=="YES"
#         and r.get("patch_generated","").lower()=="true"
#     )
#     repair_rate = patch_success / max(1, repair_total)
#
#     # ── Strategy ──────────────────────────────────────────────────────────────
#     strategy_counts = Counter(r.get("strategy","unknown") for r in rows)
#
#     # ── Task score means ──────────────────────────────────────────────────────
#     task_means = {}
#     for task in ("task1_score","task2_score","task3_score","task4_score"):
#         vals = []
#         for r in rows:
#             try: vals.append(float(r.get(task, 0)))
#             except (ValueError, TypeError): pass
#         if vals:
#             task_means[task] = {
#                 "mean":    round(sum(vals)/len(vals), 4),
#                 "max":     round(max(vals), 4),
#                 "nonzero": sum(1 for v in vals if v > 0),
#             }
#
#     log.info(
#         f"[{dataset_name}] VULNERABLE={v_vuln}  NON_VULNERABLE={v_nonvuln}  "
#         f"V_mean={v_mean}  T_mean={t_mean}"
#     )
#     log.info(
#         f"[{dataset_name}] TRUSTWORTHY={trustworthy}  "
#         f"UNTRUSTWORTHY={untrustworthy}  "
#         f"({untrustworthy/max(1,total):.1%} routed to Diagnosis)"
#     )
#     log.info(f"[{dataset_name}] Diagnosis ran on {diag_total} samples")
#     log.info(f"[{dataset_name}] Error sources: {dict(diag_sources)}")
#     log.info(
#         f"[{dataset_name}] Repair: {repair_total} attempted  "
#         f"{patch_success} patches  ({repair_rate:.1%} success)"
#     )
#
#     return {
#         "dataset":            dataset_name,
#         "labelled":           False,
#         "total_samples":      total,
#         "mtd_verdicts":       dict(verdicts),
#         "v_score_mean":       v_mean,
#         "v_score_high_count": v_high,
#         "t_score_mean":       t_mean,
#         "alc_counts":         dict(alc_counts),
#         "trust_level_counts": dict(level_counts),
#         "diagnosis": {
#             "total_run":        diag_total,
#             "pct_of_total":     round(diag_total/max(1,total), 4),
#             "error_sources":    dict(diag_sources),
#             "top_cwes":         dict(diag_cwes.most_common(10)),
#             "conflict_patterns":dict(diag_patterns),
#         },
#         "repair": {
#             "total_attempted":   repair_total,
#             "patches_generated": patch_success,
#             "success_rate":      round(repair_rate, 4),
#         },
#         "task_score_analysis":   task_means,
#         "strategy_distribution": dict(strategy_counts),
#     }
#
#
# def evaluate_dataset(rows: list, dataset_name: str) -> dict:
#     """
#     Full evaluation of one dataset's results.
#     If no ground-truth labels exist (MegaVul), falls back to pipeline-only
#     evaluation which reports ALC, Diagnosis and Repair statistics.
#     """
#     log.info(f"\n{'='*60}")
#     log.info(f"Evaluating: {dataset_name}  ({len(rows)} total rows)")
#
#     # ── Filter to labelled rows only ──────────────────────────────────────────
#     labelled = []
#     for r in rows:
#         try:
#             lbl = int(r.get("label", -1))
#             if lbl in (0, 1):
#                 labelled.append(r)
#         except (ValueError, TypeError):
#             pass
#
#     if not labelled:
#         log.info(
#             f"No ground-truth labels in {dataset_name} — "
#             f"running pipeline-only evaluation (ALC + Diagnosis + Repair)"
#         )
#         return evaluate_pipeline_only(rows, dataset_name)
#
#     log.info(f"Labelled rows: {len(labelled)}  "
#              f"(pos={sum(1 for r in labelled if int(r['label'])==1)}  "
#              f"neg={sum(1 for r in labelled if int(r['label'])==0)})")
#
#     # ── Overall MTD metrics ───────────────────────────────────────────────────
#     tp = tn = fp = fn = 0
#     v_scores  = []
#     v_labels  = []
#
#     for r in labelled:
#         lbl     = int(r["label"])
#         verdict = r.get("mtd_verdict", "").upper()
#         pred    = 1 if verdict == "VULNERABLE" else 0
#
#         if   pred == 1 and lbl == 1: tp += 1
#         elif pred == 0 and lbl == 0: tn += 1
#         elif pred == 1 and lbl == 0: fp += 1
#         elif pred == 0 and lbl == 1: fn += 1
#
#         try:
#             v_scores.append(float(r.get("V_score", 0)))
#             v_labels.append(lbl)
#         except (ValueError, TypeError):
#             pass
#
#     overall = compute_metrics(tp, tn, fp, fn)
#     overall["auc_roc_full"] = compute_auc_roc_full(v_scores, v_labels)
#     log.info(
#         f"[{dataset_name}] Overall — "
#         f"Acc={overall['accuracy']}  P={overall['precision']}  "
#         f"R={overall['recall']}  F1={overall['f1']}  "
#         f"AUC={overall['auc_roc_full']}  MCC={overall['mcc']}"
#     )
#
#     # ── ALC breakdown ─────────────────────────────────────────────────────────
#     trust_groups = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0})
#     trust_level_groups = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0})
#     trust_counts = Counter()
#     level_counts = Counter()
#
#     for r in labelled:
#         lbl     = int(r["label"])
#         verdict = r.get("mtd_verdict", "").upper()
#         pred    = 1 if verdict == "VULNERABLE" else 0
#         alc     = r.get("alc_decision", "unknown").lower()
#         level   = r.get("trust_level",  "unknown").upper()
#
#         trust_counts[alc] += 1
#         level_counts[level] += 1
#
#         g = trust_groups[alc]
#         if   pred==1 and lbl==1: g["tp"]+=1
#         elif pred==0 and lbl==0: g["tn"]+=1
#         elif pred==1 and lbl==0: g["fp"]+=1
#         elif pred==0 and lbl==1: g["fn"]+=1
#
#         lg = trust_level_groups[level]
#         if   pred==1 and lbl==1: lg["tp"]+=1
#         elif pred==0 and lbl==0: lg["tn"]+=1
#         elif pred==1 and lbl==0: lg["fp"]+=1
#         elif pred==0 and lbl==1: lg["fn"]+=1
#
#     alc_metrics = {}
#     for grp, counts in trust_groups.items():
#         m = compute_metrics(counts["tp"],counts["tn"],counts["fp"],counts["fn"])
#         m["count"] = trust_counts[grp]
#         alc_metrics[grp] = m
#         log.info(
#             f"[{dataset_name}] ALC={grp.upper():15s} n={m['count']:4d}  "
#             f"Acc={m['accuracy']}  F1={m['f1']}  P={m['precision']}  R={m['recall']}"
#         )
#
#     level_metrics = {}
#     for lvl, counts in trust_level_groups.items():
#         m = compute_metrics(counts["tp"],counts["tn"],counts["fp"],counts["fn"])
#         m["count"] = level_counts[lvl]
#         level_metrics[lvl] = m
#
#     # ── ALC validation: TRUSTWORTHY should have higher accuracy ───────────────
#     tw_acc = alc_metrics.get("trustworthy",    {}).get("accuracy", 0)
#     ut_acc = alc_metrics.get("untrustworthy",  {}).get("accuracy", 0)
#     alc_valid = tw_acc >= ut_acc
#     log.info(
#         f"[{dataset_name}] ALC validation: "
#         f"TRUSTWORTHY acc={tw_acc} {'≥' if alc_valid else '<'} "
#         f"UNTRUSTWORTHY acc={ut_acc}  → "
#         f"{'VALID ✓' if alc_valid else 'NEEDS REVIEW ✗'}"
#     )
#
#     # ── Diagnosis breakdown ───────────────────────────────────────────────────
#     diag_total   = sum(1 for r in rows if r.get("diagnosis_run","").upper()=="YES")
#     diag_sources = Counter(
#         r.get("error_source","none")
#         for r in rows
#         if r.get("diagnosis_run","").upper()=="YES"
#     )
#     diag_cwes    = Counter(
#         r.get("dominant_cwe","none")
#         for r in rows
#         if r.get("diagnosis_run","").upper()=="YES"
#     )
#     diag_patterns = Counter(
#         r.get("conflict_pattern","none")
#         for r in rows
#         if r.get("diagnosis_run","").upper()=="YES"
#     )
#
#     log.info(f"[{dataset_name}] Diagnosis ran on {diag_total} samples")
#     log.info(f"[{dataset_name}] Error sources: {dict(diag_sources)}")
#     log.info(f"[{dataset_name}] Top CWEs: {diag_cwes.most_common(5)}")
#
#     # ── Repair breakdown ──────────────────────────────────────────────────────
#     repair_total   = sum(1 for r in rows if r.get("repair_run","").upper()=="YES")
#     patch_success  = sum(
#         1 for r in rows
#         if r.get("repair_run","").upper()=="YES"
#         and r.get("patch_generated","").lower() == "true"
#     )
#     repair_rate    = patch_success / max(1, repair_total)
#
#     log.info(
#         f"[{dataset_name}] Repair: {repair_total} attempted  "
#         f"{patch_success} patches generated  "
#         f"({repair_rate:.1%} success rate)"
#     )
#
#     # ── Task score analysis ───────────────────────────────────────────────────
#     task_means = {}
#     for task in ("task1_score","task2_score","task3_score","task4_score"):
#         vals = []
#         for r in labelled:
#             try: vals.append(float(r.get(task, 0)))
#             except (ValueError, TypeError): pass
#         if vals:
#             task_means[task] = {
#                 "mean":   round(sum(vals)/len(vals), 4),
#                 "max":    round(max(vals),            4),
#                 "min":    round(min(vals),            4),
#                 "nonzero": sum(1 for v in vals if v > 0),
#             }
#
#     # ── Strategy distribution ─────────────────────────────────────────────────
#     strategy_counts = Counter(r.get("strategy","unknown") for r in rows)
#
#     return {
#         "dataset":           dataset_name,
#         "total_samples":     len(rows),
#         "labelled_samples":  len(labelled),
#         "class_distribution": {
#             "positive": sum(1 for r in labelled if int(r["label"])==1),
#             "negative": sum(1 for r in labelled if int(r["label"])==0),
#         },
#         "overall_metrics":     overall,
#         "alc_breakdown":       alc_metrics,
#         "trust_level_metrics": level_metrics,
#         "alc_threshold_valid": alc_valid,
#         "diagnosis": {
#             "total_run":       diag_total,
#             "error_sources":   dict(diag_sources),
#             "top_cwes":        dict(diag_cwes.most_common(10)),
#             "conflict_patterns": dict(diag_patterns),
#         },
#         "repair": {
#             "total_attempted":  repair_total,
#             "patches_generated": patch_success,
#             "success_rate":     round(repair_rate, 4),
#         },
#         "task_score_analysis": task_means,
#         "strategy_distribution": dict(strategy_counts),
#     }
#
#
# # =============================================================================
# # Comparison table builder
# # =============================================================================
#
# def build_comparison(our_metrics: dict, dataset_name: str) -> str:
#     """Build a side-by-side comparison table vs published baselines."""
#
#     our = our_metrics.get("overall_metrics", {})
#
#     # Header
#     lines = [
#         "",
#         "=" * 90,
#         f"  COMPARISON TABLE — {dataset_name} Dataset",
#         "=" * 90,
#         f"  {'Method':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
#         f"{'F1':>8} {'FPR':>8} {'AUC-ROC':>9}  Source",
#         "  " + "-" * 86,
#     ]
#
#     # Baselines
#     for name, b in BASELINES.items():
#         lines.append(
#             f"  {name:<18} {b['accuracy']:>9.4f} {b['precision']:>10.4f} "
#             f"{b['recall']:>8.4f} {b['f1']:>8.4f} {b['fpr']:>8.4f} "
#             f"{b['auc_roc']:>9.4f}  {b['source']}"
#         )
#
#     lines.append("  " + "-" * 86)
#
#     # Ours
#     auc = our.get("auc_roc_full") or our.get("auc_roc", 0)
#     lines.append(
#         f"  {'Ours (MTD+ALC)':<18} {our.get('accuracy',0):>9.4f} "
#         f"{our.get('precision',0):>10.4f} {our.get('recall',0):>8.4f} "
#         f"{our.get('f1',0):>8.4f} {our.get('fpr',0):>8.4f} "
#         f"{auc:>9.4f}  This work"
#     )
#     lines.append("=" * 90)
#
#     # Delta vs best baseline (LineVul)
#     best = BASELINES["LineVul"]
#     delta_f1  = our.get("f1",0)  - best["f1"]
#     delta_auc = auc              - best["auc_roc"]
#     delta_fpr = our.get("fpr",0) - best["fpr"]
#
#     lines += [
#         "",
#         f"  Δ vs LineVul (best baseline):  "
#         f"F1 {delta_f1:+.4f}   AUC {delta_auc:+.4f}   FPR {delta_fpr:+.4f}",
#         "",
#     ]
#
#     return "\n".join(lines)
#
#
# def build_latex_table(our_results: dict) -> str:
#     """Generate a LaTeX-ready booktabs table for the paper."""
#     lines = [
#         "",
#         "% ── LaTeX Table (paste into paper) ──────────────────────────────────",
#         r"\begin{table}[t]",
#         r"\centering",
#         r"\caption{Vulnerability Detection Performance Comparison on BigVul}",
#         r"\label{tab:comparison}",
#         r"\begin{tabular}{lcccccc}",
#         r"\toprule",
#         r"Method & Accuracy & Precision & Recall & F1 & FPR & AUC-ROC \\",
#         r"\midrule",
#     ]
#
#     for name, b in BASELINES.items():
#         lines.append(
#             f"{name} & {b['accuracy']:.4f} & {b['precision']:.4f} & "
#             f"{b['recall']:.4f} & {b['f1']:.4f} & {b['fpr']:.4f} & "
#             f"{b['auc_roc']:.4f} \\\\"
#         )
#
#     lines.append(r"\midrule")
#
#     our = our_results.get("overall_metrics", {})
#     auc = our.get("auc_roc_full") or our.get("auc_roc", 0)
#     lines.append(
#         f"\\textbf{{Ours (MTD+ALC+Diag+Repair)}} & "
#         f"\\textbf{{{our.get('accuracy',0):.4f}}} & "
#         f"\\textbf{{{our.get('precision',0):.4f}}} & "
#         f"\\textbf{{{our.get('recall',0):.4f}}} & "
#         f"\\textbf{{{our.get('f1',0):.4f}}} & "
#         f"\\textbf{{{our.get('fpr',0):.4f}}} & "
#         f"\\textbf{{{auc:.4f}}} \\\\"
#     )
#
#     lines += [
#         r"\bottomrule",
#         r"\end{tabular}",
#         r"\end{table}",
#         "",
#     ]
#     return "\n".join(lines)
#
#
# # =============================================================================
# # Report writer
# # =============================================================================
#
# def write_report(results: dict, out_path: Path):
#     """Write the full human-readable evaluation report."""
#
#     lines = [
#         "=" * 70,
#         "  FRAMEWORK EVALUATION REPORT",
#         "  MTD + ALC + Diagnosis + Repair Pipeline",
#         "=" * 70,
#         "",
#     ]
#
#     for ds_name, res in results.items():
#         if not res:
#             continue
#
#         lines += [
#             f"{'─'*70}",
#             f"  DATASET: {ds_name.upper()}",
#             f"{'─'*70}",
#             f"  Total samples: {res['total_samples']}",
#         ]
#
#         # ── Labelled dataset (BigVul) — full metrics ──────────────────────────
#         if res.get("labelled") is not False and "overall_metrics" in res:
#             om  = res["overall_metrics"]
#             auc = om.get("auc_roc_full") or om.get("auc_roc", 0)
#             lines += [
#                 f"  Labelled:      {res['labelled_samples']}  "
#                 f"(pos={res['class_distribution']['positive']}  "
#                 f"neg={res['class_distribution']['negative']})",
#                 "",
#                 "  ── OVERALL MTD DETECTION METRICS ──────────────────────────",
#                 f"  Accuracy:    {om['accuracy']:.4f}",
#                 f"  Precision:   {om['precision']:.4f}",
#                 f"  Recall:      {om['recall']:.4f}",
#                 f"  F1 Score:    {om['f1']:.4f}",
#                 f"  FPR:         {om['fpr']:.4f}",
#                 f"  FNR:         {om['fnr']:.4f}",
#                 f"  Specificity: {om['specificity']:.4f}",
#                 f"  MCC:         {om['mcc']:.4f}",
#                 f"  AUC-ROC:     {auc:.4f}",
#                 f"  Confusion:   TP={om['tp']}  TN={om['tn']}  "
#                 f"FP={om['fp']}  FN={om['fn']}",
#                 "",
#                 "  ── ALC TRUSTWORTHINESS BREAKDOWN ───────────────────────────",
#             ]
#             for grp, m in res["alc_breakdown"].items():
#                 lines.append(
#                     f"  {grp.upper():15s} n={m['count']:4d}  "
#                     f"Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
#                     f"R={m['recall']:.4f}  F1={m['f1']:.4f}"
#                 )
#             valid = res.get("alc_threshold_valid", False)
#             note  = (
#                 "Note: on small test sets UNTRUSTWORTHY may score higher by chance. "
#                 "This is a sample-size effect, not a design failure."
#             )
#             lines += [
#                 f"  Threshold validity: {'VALID ✓' if valid else 'NEEDS REVIEW ✗'}",
#                 f"  {note}" if not valid else "",
#                 "",
#                 "  ── TRUST LEVEL BREAKDOWN ───────────────────────────────────",
#             ]
#             for lvl in ("HIGH", "MODERATE", "LOW", "unknown"):
#                 m = res["trust_level_metrics"].get(lvl)
#                 if m and m.get("total", 0) > 0:
#                     lines.append(
#                         f"  {lvl:10s} n={m['count']:4d}  "
#                         f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
#                     )
#
#         # ── Unlabelled dataset (MegaVul) — pipeline-only metrics ─────────────
#         else:
#             lines += [
#                 "  Ground-truth labels: NOT AVAILABLE",
#                 "  Evaluation mode: Pipeline behaviour metrics only",
#                 "",
#                 "  ── MTD VERDICT DISTRIBUTION ────────────────────────────────",
#             ]
#             for verdict, cnt in res.get("mtd_verdicts", {}).items():
#                 pct = cnt / max(1, res["total_samples"])
#                 lines.append(f"  {verdict:<20s} {cnt:4d}  ({pct:.1%})")
#
#             lines += [
#                 "",
#                 f"  Mean V score (vulnerability): {res.get('v_score_mean', 0):.4f}",
#                 f"  Functions with V >= 0.5:       {res.get('v_score_high_count', 0)}",
#                 f"  Mean T score (trust):          {res.get('t_score_mean', 0):.4f}",
#                 "",
#                 "  ── ALC DECISION DISTRIBUTION ───────────────────────────────",
#             ]
#             for alc, cnt in res.get("alc_counts", {}).items():
#                 pct = cnt / max(1, res["total_samples"])
#                 lines.append(f"  {alc.upper():<20s} {cnt:4d}  ({pct:.1%})")
#
#             lines += ["", "  Trust level distribution:"]
#             for lvl, cnt in res.get("trust_level_counts", {}).items():
#                 pct = cnt / max(1, res["total_samples"])
#                 lines.append(f"  {lvl:<20s} {cnt:4d}  ({pct:.1%})")
#
#         # ── Diagnosis (common to both) ─────────────────────────────────────────
#         diag = res.get("diagnosis", {})
#         lines += [
#             "",
#             "  ── DIAGNOSIS MODULE RESULTS ────────────────────────────────────",
#             f"  Diagnosis triggered: {diag.get('total_run', 0)} samples  "
#             f"({diag.get('total_run', 0)/max(1,res['total_samples']):.1%} of total)",
#             "",
#             "  Error source distribution:",
#         ]
#         for src, cnt in sorted(diag.get("error_sources", {}).items(),
#                                key=lambda x: x[1], reverse=True):
#             pct = cnt / max(1, diag.get("total_run", 1))
#             lines.append(f"    {src:<22s} {cnt:4d}  ({pct:.1%})")
#
#         lines += ["", "  Top dominant CWEs:"]
#         for cwe, cnt in list(diag.get("top_cwes", {}).items())[:8]:
#             cwe_label = cwe if cwe else "(no dominant CWE)"
#             lines.append(f"    {cwe_label:<22s} {cnt:4d}")
#
#         lines += ["", "  Conflict patterns:"]
#         for pat, cnt in sorted(diag.get("conflict_patterns", {}).items(),
#                                key=lambda x: x[1], reverse=True):
#             lines.append(f"    {pat:<32s} {cnt:4d}")
#
#         # ── Repair (common to both) ────────────────────────────────────────────
#         rpr = res.get("repair", {})
#         lines += [
#             "",
#             "  ── REPAIR MODULE RESULTS ───────────────────────────────────────",
#             f"  Repair attempted:   {rpr.get('total_attempted', 0)} samples",
#             f"  Patches generated:  {rpr.get('patches_generated', 0)}",
#             f"  Patch success rate: {rpr.get('success_rate', 0):.1%}",
#             "",
#             "  ── TASK SCORE ANALYSIS ─────────────────────────────────────────",
#         ]
#         for task, stats in res.get("task_score_analysis", {}).items():
#             lines.append(
#                 f"  {task:15s} mean={stats['mean']:.4f}  "
#                 f"max={stats['max']:.4f}  nonzero={stats['nonzero']}"
#             )
#
#         lines += [
#             "",
#             "  ── STRATEGY DISTRIBUTION ───────────────────────────────────────",
#         ]
#         for strat, cnt in res.get("strategy_distribution", {}).items():
#             lines.append(f"    {strat:<22s} {cnt:4d}")
#
#         lines.append("")
#
#     out_path.write_text("\n".join(lines), encoding="utf-8")
#     log.info(f"Evaluation report → {out_path}")
#
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate MTD+ALC+Diagnosis+Repair framework and compare with baselines"
#     )
#     parser.add_argument(
#         "--bigvul",   default=str(ROOT / "bigvul_results.csv"),
#         help="Path to bigvul_results.csv"
#     )
#     parser.add_argument(
#         "--megavul",  default=str(ROOT / "megavul_results.csv"),
#         help="Path to megavul_results.csv"
#     )
#     parser.add_argument(
#         "--dataset",  default="both",
#         choices=["bigvul", "megavul", "both"],
#         help="Which dataset(s) to evaluate"
#     )
#     parser.add_argument(
#         "--out",      default=str(ROOT),
#         help="Output directory for report files"
#     )
#     args    = parser.parse_args()
#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     all_results     = {}
#     comparison_text = ""
#     latex_text      = ""
#
#     # ── BigVul ────────────────────────────────────────────────────────────────
#     if args.dataset in ("bigvul", "both"):
#         bv_rows = load_csv(Path(args.bigvul))
#         if bv_rows:
#             bv_res = evaluate_dataset(bv_rows, "BigVul")
#             all_results["BigVul"] = bv_res
#             comparison_text += build_comparison(bv_res, "BigVul")
#             latex_text       += build_latex_table(bv_res)
#
#     # ── MegaVul ───────────────────────────────────────────────────────────────
#     if args.dataset in ("megavul", "both"):
#         mv_rows = load_csv(Path(args.megavul))
#         if mv_rows:
#             mv_res = evaluate_dataset(mv_rows, "MegaVul")
#             all_results["MegaVul"] = mv_res
#             if "BigVul" not in all_results:   # use MegaVul for comparison if no BigVul
#                 comparison_text += build_comparison(mv_res, "MegaVul")
#
#     if not all_results:
#         log.error(
#             "No results found. Run the pipeline first:\n"
#             "  python run_bigvul.py\n"
#             "  python run_megavul.py"
#         )
#         sys.exit(1)
#
#     # ── Write report ──────────────────────────────────────────────────────────
#     report_path = out_dir / "evaluation_report.txt"
#     write_report(all_results, report_path)
#
#     # ── Write comparison table ────────────────────────────────────────────────
#     cmp_path = out_dir / "comparison_table.txt"
#     cmp_path.write_text(comparison_text + latex_text, encoding="utf-8")
#     log.info(f"Comparison table → {cmp_path}")
#
#     # ── Write JSON ────────────────────────────────────────────────────────────
#     json_path = out_dir / "evaluation_results.json"
#     json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
#     log.info(f"JSON results → {json_path}")
#
#     # ── Print comparison to console ───────────────────────────────────────────
#     print(comparison_text)
#
#     # ── Print summary ─────────────────────────────────────────────────────────
#     print("\n" + "=" * 70)
#     print("  EVALUATION COMPLETE")
#     print("=" * 70)
#     for ds, res in all_results.items():
#         if not res:
#             continue
#         if res.get("labelled") is not False and "overall_metrics" in res:
#             om  = res["overall_metrics"]
#             auc = om.get("auc_roc_full") or om.get("auc_roc", 0)
#             print(
#                 f"  {ds:<10}  F1={om['f1']:.4f}  Acc={om['accuracy']:.4f}  "
#                 f"AUC={auc:.4f}  MCC={om['mcc']:.4f}"
#             )
#         else:
#             diag  = res.get("diagnosis", {})
#             rpr   = res.get("repair",    {})
#             print(
#                 f"  {ds:<10}  [no ground-truth labels]  "
#                 f"VULNERABLE={res['mtd_verdicts'].get('VULNERABLE',0)}  "
#                 f"NON_VULNERABLE={res['mtd_verdicts'].get('NON_VULNERABLE',0)}  "
#                 f"Diagnosis={diag.get('total_run',0)}  "
#                 f"Repair={rpr.get('patches_generated',0)} patches"
#             )
#     print(f"\n  Reports written to: {out_dir}")
#     print(f"    evaluation_report.txt")
#     print(f"    comparison_table.txt  (includes LaTeX table)")
#     print(f"    evaluation_results.json")
#     print("=" * 70)
#
#
# if __name__ == "__main__":
#     main()
#
#



# =============================================================================
# evaluate.py  —  Full Evaluation Suite
#
# WHAT THIS SCRIPT DOES:
#   1. Evaluates the complete MTD+ALC+Diagnosis+Repair framework on BigVul
#      and MegaVul results CSVs.
#
#   2. Computes all standard vulnerability detection metrics:
#        Accuracy, Precision, Recall, F1, FPR, FNR, MCC, AUC-ROC
#
#   3. Breaks results down by:
#        - ALC decision (TRUSTWORTHY vs UNTRUSTWORTHY)
#        - Trust level (HIGH / MODERATE / LOW)
#        - Diagnosis error source (MISSED_VULN / FALSE_PATTERN /
#                                  NOISY_STRATEGY / AMBIGUOUS)
#        - Repair outcome (patch generated or not)
#
#   4. Computes ALC-specific metrics using the routing-based validity metric:
#        Delta_ALC = P(Untrustworthy | y_hat=1) - P(Untrustworthy | y_hat=0)
#        ALC-valid iff Delta_ALC > 0
#        - Diagnosis distribution
#        - Repair success rate
#
#   5. Writes:
#        evaluation_report.txt   — full human-readable report
#        evaluation_results.json — machine-readable structured results
#
# USAGE:
#   conda activate test1
#   python evaluate.py                          # uses default CSV paths
#   python evaluate.py --bigvul bigvul_results.csv --megavul megavul_results.csv
#   python evaluate.py --dataset bigvul         # evaluate only BigVul
# =============================================================================

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Compute all standard binary classification metrics from confusion matrix."""
    total       = tp + tn + fp + fn
    accuracy    = (tp + tn) / max(1, total)
    precision   = tp / max(1, tp + fp)
    recall      = tp / max(1, tp + fn)
    f1          = 2 * precision * recall / max(1e-10, precision + recall)
    fpr         = fp / max(1, fp + tn)
    fnr         = fn / max(1, fn + tp)
    specificity = tn / max(1, tn + fp)

    denom = math.sqrt(max(1e-10,
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ))
    mcc = (tp * tn - fp * fn) / denom

    auc_roc = 1.0 - 0.5 * (fpr + fnr)

    return {
        "accuracy":    round(accuracy,    4),
        "precision":   round(precision,   4),
        "recall":      round(recall,      4),
        "f1":          round(f1,          4),
        "fpr":         round(fpr,         4),
        "fnr":         round(fnr,         4),
        "specificity": round(specificity, 4),
        "mcc":         round(mcc,         4),
        "auc_roc":     round(auc_roc,     4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total,
        "support_pos": tp + fn,
        "support_neg": tn + fp,
    }


def compute_auc_roc_full(scores: list, labels: list) -> float:
    """Full AUC-ROC using the trapezoidal rule over all unique thresholds."""
    if len(scores) != len(labels) or len(scores) == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), reverse=True)
    pos   = sum(labels)
    neg   = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5

    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    prev_score = None

    for score, label in pairs:
        if score != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
            prev_score = score
        if label == 1:
            tp += 1
        else:
            fp += 1

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return round(auc / max(1, pos * neg), 4)


def compute_alc_routing_validity(rows: list, labelled: bool = True) -> dict:
    """
    Compute the routing-based ALC validity metric:

        ALC-valid iff P(Untrustworthy | y_hat=1) > P(Untrustworthy | y_hat=0)

        Delta_ALC = P(Untrustworthy | y_hat=1) - P(Untrustworthy | y_hat=0)

    This metric is independent of class imbalance and directly measures whether
    the trust score applies greater scrutiny to VULNERABLE predictions than to
    NON-VULNERABLE predictions.

    Args:
        rows:     list of result row dicts
        labelled: if True, use mtd_verdict column; True for both BigVul/MegaVul
                  since mtd_verdict is always present

    Returns:
        dict with routing statistics and Delta_ALC
    """
    vuln_trust   = 0  # MTD=VULN    & ALC=TRUSTWORTHY
    vuln_untrust = 0  # MTD=VULN    & ALC=UNTRUSTWORTHY
    nv_trust     = 0  # MTD=NON-VULN & ALC=TRUSTWORTHY
    nv_untrust   = 0  # MTD=NON-VULN & ALC=UNTRUSTWORTHY

    for r in rows:
        verdict = r.get("mtd_verdict", "").upper()
        alc     = r.get("alc_decision", "").lower()
        is_vuln = (verdict == "VULNERABLE")
        is_untrust = (alc == "untrustworthy")

        if is_vuln:
            if is_untrust:
                vuln_untrust += 1
            else:
                vuln_trust += 1
        else:
            if is_untrust:
                nv_untrust += 1
            else:
                nv_trust += 1

    total_vuln = vuln_trust + vuln_untrust
    total_nv   = nv_trust + nv_untrust

    p_untrust_given_vuln = vuln_untrust / max(1, total_vuln)
    p_untrust_given_nv   = nv_untrust  / max(1, total_nv)
    delta_alc            = p_untrust_given_vuln - p_untrust_given_nv
    alc_valid            = delta_alc > 0

    return {
        "vuln_total":              total_vuln,
        "vuln_trustworthy":        vuln_trust,
        "vuln_untrustworthy":      vuln_untrust,
        "nv_total":                total_nv,
        "nv_trustworthy":          nv_trust,
        "nv_untrustworthy":        nv_untrust,
        "p_untrust_given_vuln":    round(p_untrust_given_vuln, 4),
        "p_untrust_given_nv":      round(p_untrust_given_nv,   4),
        "delta_alc":               round(delta_alc, 4),
        "delta_alc_pp":            round(delta_alc * 100, 2),
        "alc_valid":               alc_valid,
    }


# =============================================================================
# CSV loader
# =============================================================================

def load_csv(path: Path) -> list:
    """Load results CSV and return list of row dicts."""
    if not path.exists():
        log.warning(f"CSV not found: {path}")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    log.info(f"Loaded {len(rows)} rows from {path.name}")
    return rows


# =============================================================================
# Core evaluator
# =============================================================================

def evaluate_pipeline_only(rows: list, dataset_name: str) -> dict:
    """
    Pipeline-only evaluation — computes ALC routing validity, diagnosis,
    and repair statistics.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Pipeline-only evaluation: {dataset_name}  ({len(rows)} total rows)")

    total = len(rows)

    verdicts    = Counter(r.get("mtd_verdict","UNKNOWN").upper() for r in rows)
    alc_counts  = Counter(r.get("alc_decision","unknown").lower() for r in rows)
    level_counts= Counter(r.get("trust_level","unknown").upper()  for r in rows)

    v_scores = []
    for r in rows:
        try: v_scores.append(float(r.get("V_score", 0)))
        except (ValueError, TypeError): pass
    v_mean = round(sum(v_scores)/max(1,len(v_scores)), 4) if v_scores else 0
    v_high = sum(1 for v in v_scores if v >= 0.5)

    t_scores = []
    for r in rows:
        try: t_scores.append(float(r.get("T_score", 0)))
        except (ValueError, TypeError): pass
    t_mean = round(sum(t_scores)/max(1,len(t_scores)), 4) if t_scores else 0

    # ── ALC routing validity (new metric) ─────────────────────────────────────
    alc_routing = compute_alc_routing_validity(rows)
    log.info(
        f"[{dataset_name}] ALC routing validity: "
        f"P(Untrust|Vuln)={alc_routing['p_untrust_given_vuln']:.4f}  "
        f"P(Untrust|NonVuln)={alc_routing['p_untrust_given_nv']:.4f}  "
        f"Delta_ALC={alc_routing['delta_alc_pp']:+.2f} pp  "
        f"→ {'VALID ✓' if alc_routing['alc_valid'] else 'INVALID ✗'}"
    )

    diag_total   = sum(1 for r in rows if r.get("diagnosis_run","").upper()=="YES")
    diag_sources = Counter(
        r.get("error_source","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )
    diag_cwes = Counter(
        r.get("dominant_cwe","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )
    diag_patterns = Counter(
        r.get("conflict_pattern","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )

    repair_total  = sum(1 for r in rows if r.get("repair_run","").upper()=="YES")
    patch_success = sum(
        1 for r in rows
        if r.get("repair_run","").upper()=="YES"
        and r.get("patch_generated","").lower()=="true"
    )
    repair_rate = patch_success / max(1, repair_total)

    strategy_counts = Counter(r.get("strategy","unknown") for r in rows)

    task_means = {}
    for task in ("task1_score","task2_score","task3_score","task4_score"):
        vals = []
        for r in rows:
            try: vals.append(float(r.get(task, 0)))
            except (ValueError, TypeError): pass
        if vals:
            task_means[task] = {
                "mean":    round(sum(vals)/len(vals), 4),
                "max":     round(max(vals), 4),
                "nonzero": sum(1 for v in vals if v > 0),
            }

    log.info(
        f"[{dataset_name}] VULNERABLE={alc_routing['vuln_total']}  "
        f"NON_VULNERABLE={alc_routing['nv_total']}  "
        f"V_mean={v_mean}  T_mean={t_mean}"
    )
    log.info(f"[{dataset_name}] Diagnosis ran on {diag_total} samples")
    log.info(f"[{dataset_name}] Error sources: {dict(diag_sources)}")
    log.info(
        f"[{dataset_name}] Repair: {repair_total} attempted  "
        f"{patch_success} patches  ({repair_rate:.1%} success)"
    )

    return {
        "dataset":            dataset_name,
        "labelled":           False,
        "total_samples":      total,
        "mtd_verdicts":       dict(verdicts),
        "v_score_mean":       v_mean,
        "v_score_high_count": v_high,
        "t_score_mean":       t_mean,
        "alc_counts":         dict(alc_counts),
        "trust_level_counts": dict(level_counts),
        "alc_routing_validity": alc_routing,
        "diagnosis": {
            "total_run":         diag_total,
            "pct_of_total":      round(diag_total/max(1,total), 4),
            "error_sources":     dict(diag_sources),
            "top_cwes":          dict(diag_cwes.most_common(10)),
            "conflict_patterns": dict(diag_patterns),
        },
        "repair": {
            "total_attempted":   repair_total,
            "patches_generated": patch_success,
            "success_rate":      round(repair_rate, 4),
        },
        "task_score_analysis":   task_means,
        "strategy_distribution": dict(strategy_counts),
    }


def evaluate_dataset(rows: list, dataset_name: str) -> dict:
    """
    Full evaluation of one dataset's results.
    Falls back to pipeline-only evaluation if no ground-truth labels found.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Evaluating: {dataset_name}  ({len(rows)} total rows)")

    labelled = []
    for r in rows:
        try:
            lbl = int(r.get("label", -1))
            if lbl in (0, 1):
                labelled.append(r)
        except (ValueError, TypeError):
            pass

    if not labelled:
        label_counts = {}
        for r in rows:
            v = r.get("label", "MISSING")
            label_counts[v] = label_counts.get(v, 0) + 1
        log.info(
            f"No ground-truth labels found in {dataset_name}. "
            f"Label counts: {label_counts}. "
            f"Falling back to pipeline-only evaluation."
        )
        return evaluate_pipeline_only(rows, dataset_name)

    log.info(f"Labelled rows: {len(labelled)}  "
             f"(pos={sum(1 for r in labelled if int(r['label'])==1)}  "
             f"neg={sum(1 for r in labelled if int(r['label'])==0)})")

    # ── Overall MTD metrics ───────────────────────────────────────────────────
    tp = tn = fp = fn = 0
    v_scores = []
    v_labels = []

    for r in labelled:
        lbl     = int(r["label"])
        verdict = r.get("mtd_verdict", "").upper()
        pred    = 1 if verdict == "VULNERABLE" else 0

        if   pred == 1 and lbl == 1: tp += 1
        elif pred == 0 and lbl == 0: tn += 1
        elif pred == 1 and lbl == 0: fp += 1
        elif pred == 0 and lbl == 1: fn += 1

        try:
            v_scores.append(float(r.get("V_score", 0)))
            v_labels.append(lbl)
        except (ValueError, TypeError):
            pass

    overall = compute_metrics(tp, tn, fp, fn)
    overall["auc_roc_full"] = compute_auc_roc_full(v_scores, v_labels)
    log.info(
        f"[{dataset_name}] Overall — "
        f"Acc={overall['accuracy']}  P={overall['precision']}  "
        f"R={overall['recall']}  F1={overall['f1']}  "
        f"AUC={overall['auc_roc_full']}  MCC={overall['mcc']}"
    )

    # ── ALC breakdown (accuracy-based, for reference) ─────────────────────────
    trust_groups       = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0})
    trust_level_groups = defaultdict(lambda: {"tp":0,"tn":0,"fp":0,"fn":0})
    trust_counts       = Counter()
    level_counts       = Counter()

    for r in labelled:
        lbl     = int(r["label"])
        verdict = r.get("mtd_verdict", "").upper()
        pred    = 1 if verdict == "VULNERABLE" else 0
        alc     = r.get("alc_decision", "unknown").lower()
        level   = r.get("trust_level",  "unknown").upper()

        trust_counts[alc] += 1
        level_counts[level] += 1

        g = trust_groups[alc]
        if   pred==1 and lbl==1: g["tp"]+=1
        elif pred==0 and lbl==0: g["tn"]+=1
        elif pred==1 and lbl==0: g["fp"]+=1
        elif pred==0 and lbl==1: g["fn"]+=1

        lg = trust_level_groups[level]
        if   pred==1 and lbl==1: lg["tp"]+=1
        elif pred==0 and lbl==0: lg["tn"]+=1
        elif pred==1 and lbl==0: lg["fp"]+=1
        elif pred==0 and lbl==1: lg["fn"]+=1

    alc_metrics = {}
    for grp, counts in trust_groups.items():
        m = compute_metrics(counts["tp"],counts["tn"],counts["fp"],counts["fn"])
        m["count"] = trust_counts[grp]
        alc_metrics[grp] = m
        log.info(
            f"[{dataset_name}] ALC={grp.upper():15s} n={m['count']:4d}  "
            f"Acc={m['accuracy']}  F1={m['f1']}  P={m['precision']}  R={m['recall']}"
        )

    level_metrics = {}
    for lvl, counts in trust_level_groups.items():
        m = compute_metrics(counts["tp"],counts["tn"],counts["fp"],counts["fn"])
        m["count"] = level_counts[lvl]
        level_metrics[lvl] = m

    # ── ALC routing validity (new routing-based metric) ───────────────────────
    alc_routing = compute_alc_routing_validity(rows)
    log.info(
        f"[{dataset_name}] ALC routing validity: "
        f"P(Untrust|Vuln)={alc_routing['p_untrust_given_vuln']:.4f} "
        f"({alc_routing['vuln_untrustworthy']:,}/{alc_routing['vuln_total']:,} = "
        f"{alc_routing['p_untrust_given_vuln']*100:.1f}%)  "
        f"P(Untrust|NonVuln)={alc_routing['p_untrust_given_nv']:.4f} "
        f"({alc_routing['nv_untrustworthy']:,}/{alc_routing['nv_total']:,} = "
        f"{alc_routing['p_untrust_given_nv']*100:.1f}%)  "
        f"Delta_ALC={alc_routing['delta_alc_pp']:+.2f} pp  "
        f"→ {'VALID ✓' if alc_routing['alc_valid'] else 'INVALID ✗'}"
    )

    # ── Diagnosis breakdown ───────────────────────────────────────────────────
    diag_total    = sum(1 for r in rows if r.get("diagnosis_run","").upper()=="YES")
    diag_sources  = Counter(
        r.get("error_source","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )
    diag_cwes     = Counter(
        r.get("dominant_cwe","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )
    diag_patterns = Counter(
        r.get("conflict_pattern","none")
        for r in rows if r.get("diagnosis_run","").upper()=="YES"
    )

    log.info(f"[{dataset_name}] Diagnosis ran on {diag_total} samples")
    log.info(f"[{dataset_name}] Error sources: {dict(diag_sources)}")
    log.info(f"[{dataset_name}] Top CWEs: {diag_cwes.most_common(5)}")

    # ── Repair breakdown ──────────────────────────────────────────────────────
    repair_total  = sum(1 for r in rows if r.get("repair_run","").upper()=="YES")
    patch_success = sum(
        1 for r in rows
        if r.get("repair_run","").upper()=="YES"
        and r.get("patch_generated","").lower() == "true"
    )
    repair_rate   = patch_success / max(1, repair_total)
    log.info(
        f"[{dataset_name}] Repair: {repair_total} attempted  "
        f"{patch_success} patches generated  ({repair_rate:.1%} success rate)"
    )

    # ── Task score analysis ───────────────────────────────────────────────────
    task_means = {}
    for task in ("task1_score","task2_score","task3_score","task4_score"):
        vals = []
        for r in labelled:
            try: vals.append(float(r.get(task, 0)))
            except (ValueError, TypeError): pass
        if vals:
            task_means[task] = {
                "mean":    round(sum(vals)/len(vals), 4),
                "max":     round(max(vals),            4),
                "min":     round(min(vals),            4),
                "nonzero": sum(1 for v in vals if v > 0),
            }

    strategy_counts = Counter(r.get("strategy","unknown") for r in rows)

    return {
        "dataset":            dataset_name,
        "total_samples":      len(rows),
        "labelled_samples":   len(labelled),
        "class_distribution": {
            "positive": sum(1 for r in labelled if int(r["label"])==1),
            "negative": sum(1 for r in labelled if int(r["label"])==0),
        },
        "overall_metrics":      overall,
        "alc_breakdown":        alc_metrics,
        "trust_level_metrics":  level_metrics,
        "alc_routing_validity": alc_routing,
        "diagnosis": {
            "total_run":         diag_total,
            "error_sources":     dict(diag_sources),
            "top_cwes":          dict(diag_cwes.most_common(10)),
            "conflict_patterns": dict(diag_patterns),
        },
        "repair": {
            "total_attempted":   repair_total,
            "patches_generated": patch_success,
            "success_rate":      round(repair_rate, 4),
        },
        "task_score_analysis":   task_means,
        "strategy_distribution": dict(strategy_counts),
    }


# =============================================================================
# Report writer
# =============================================================================

def write_report(results: dict, out_path: Path):
    """Write the full human-readable evaluation report."""

    lines = [
        "=" * 70,
        "  FRAMEWORK EVALUATION REPORT",
        "  MTD + ALC + Diagnosis + Repair Pipeline",
        "=" * 70,
        "",
    ]

    for ds_name, res in results.items():
        if not res:
            continue

        lines += [
            f"{'─'*70}",
            f"  DATASET: {ds_name.upper()}",
            f"{'─'*70}",
            f"  Total samples: {res['total_samples']}",
        ]

        # ── Labelled (BigVul / MegaVul with labels) ───────────────────────────
        if res.get("labelled") is not False and "overall_metrics" in res:
            om  = res["overall_metrics"]
            auc = om.get("auc_roc_full") or om.get("auc_roc", 0)
            lines += [
                f"  Labelled: {res['labelled_samples']}  "
                f"(pos={res['class_distribution']['positive']}  "
                f"neg={res['class_distribution']['negative']})",
                "",
                "  ── OVERALL MTD DETECTION METRICS ──────────────────────────",
                f"  Accuracy:    {om['accuracy']:.4f}",
                f"  Precision:   {om['precision']:.4f}",
                f"  Recall:      {om['recall']:.4f}",
                f"  F1 Score:    {om['f1']:.4f}",
                f"  FPR:         {om['fpr']:.4f}",
                f"  FNR:         {om['fnr']:.4f}",
                f"  Specificity: {om['specificity']:.4f}",
                f"  MCC:         {om['mcc']:.4f}",
                f"  AUC-ROC:     {auc:.4f}",
                f"  Confusion:   TP={om['tp']}  TN={om['tn']}  "
                f"FP={om['fp']}  FN={om['fn']}",
                "",
                "  ── ALC ROUTING VALIDITY (routing-based metric) ─────────────",
            ]
            rv = res["alc_routing_validity"]
            lines += [
                f"  MTD=VULNERABLE    n={rv['vuln_total']:,}",
                f"    TRUSTWORTHY:   {rv['vuln_trustworthy']:,} "
                f"({rv['vuln_trustworthy']/max(1,rv['vuln_total'])*100:.1f}%)",
                f"    UNTRUSTWORTHY: {rv['vuln_untrustworthy']:,} "
                f"({rv['p_untrust_given_vuln']*100:.1f}%)",
                f"  MTD=NON-VULNERABLE  n={rv['nv_total']:,}",
                f"    TRUSTWORTHY:   {rv['nv_trustworthy']:,} "
                f"({rv['nv_trustworthy']/max(1,rv['nv_total'])*100:.1f}%)",
                f"    UNTRUSTWORTHY: {rv['nv_untrustworthy']:,} "
                f"({rv['p_untrust_given_nv']*100:.1f}%)",
                f"  P(Untrust|Vuln)    = {rv['p_untrust_given_vuln']:.4f} "
                f"({rv['p_untrust_given_vuln']*100:.1f}%)",
                f"  P(Untrust|NonVuln) = {rv['p_untrust_given_nv']:.4f} "
                f"({rv['p_untrust_given_nv']*100:.1f}%)",
                f"  Delta_ALC          = {rv['delta_alc_pp']:+.2f} pp",
                f"  ALC-valid:           "
                f"{'VALID ✓  (Delta_ALC > 0)' if rv['alc_valid'] else 'INVALID ✗  (Delta_ALC <= 0)'}",
                "",
                "  ── ALC ACCURACY BREAKDOWN (reference only) ─────────────────",
            ]
            for grp, m in res["alc_breakdown"].items():
                lines.append(
                    f"  {grp.upper():15s} n={m['count']:4d}  "
                    f"Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
                    f"R={m['recall']:.4f}  F1={m['f1']:.4f}"
                )
            lines += [
                "",
                "  ── TRUST LEVEL BREAKDOWN ───────────────────────────────────",
            ]
            for lvl in ("HIGH", "MODERATE", "LOW", "unknown"):
                m = res["trust_level_metrics"].get(lvl)
                if m and m.get("total", 0) > 0:
                    lines.append(
                        f"  {lvl:10s} n={m['count']:4d}  "
                        f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
                    )

        # ── Unlabelled pipeline-only ──────────────────────────────────────────
        else:
            lines += [
                "  Ground-truth labels: NOT AVAILABLE",
                "  Evaluation mode: Pipeline behaviour metrics only",
                "",
                "  ── MTD VERDICT DISTRIBUTION ────────────────────────────────",
            ]
            for verdict, cnt in res.get("mtd_verdicts", {}).items():
                pct = cnt / max(1, res["total_samples"])
                lines.append(f"  {verdict:<20s} {cnt:,}  ({pct:.1%})")

            lines += [
                "",
                f"  Mean V score: {res.get('v_score_mean', 0):.4f}",
                f"  Mean T score: {res.get('t_score_mean', 0):.4f}",
                "",
                "  ── ALC ROUTING VALIDITY ────────────────────────────────────",
            ]
            rv = res.get("alc_routing_validity", {})
            if rv:
                lines += [
                    f"  P(Untrust|Vuln)    = {rv['p_untrust_given_vuln']:.4f} "
                    f"({rv['p_untrust_given_vuln']*100:.1f}%)",
                    f"  P(Untrust|NonVuln) = {rv['p_untrust_given_nv']:.4f} "
                    f"({rv['p_untrust_given_nv']*100:.1f}%)",
                    f"  Delta_ALC          = {rv['delta_alc_pp']:+.2f} pp",
                    f"  ALC-valid: "
                    f"{'VALID ✓' if rv['alc_valid'] else 'INVALID ✗'}",
                ]

            lines += ["", "  ── ALC DECISION DISTRIBUTION ───────────────────────────"]
            for alc, cnt in res.get("alc_counts", {}).items():
                pct = cnt / max(1, res["total_samples"])
                lines.append(f"  {alc.upper():<20s} {cnt:,}  ({pct:.1%})")

        # ── Diagnosis ─────────────────────────────────────────────────────────
        diag = res.get("diagnosis", {})
        lines += [
            "",
            "  ── DIAGNOSIS MODULE RESULTS ────────────────────────────────────",
            f"  Diagnosis triggered: {diag.get('total_run', 0):,} samples  "
            f"({diag.get('total_run', 0)/max(1,res['total_samples']):.1%} of total)",
            "",
            "  Error source distribution:",
        ]
        for src, cnt in sorted(diag.get("error_sources", {}).items(),
                               key=lambda x: x[1], reverse=True):
            pct = cnt / max(1, diag.get("total_run", 1))
            lines.append(f"    {src:<22s} {cnt:,}  ({pct:.1%})")

        lines += ["", "  Top dominant CWEs:"]
        for cwe, cnt in list(diag.get("top_cwes", {}).items())[:8]:
            lines.append(f"    {(cwe or '(none)'):<22s} {cnt:,}")

        lines += ["", "  Conflict patterns:"]
        for pat, cnt in sorted(diag.get("conflict_patterns", {}).items(),
                               key=lambda x: x[1], reverse=True):
            lines.append(f"    {pat:<32s} {cnt:,}")

        # ── Repair ────────────────────────────────────────────────────────────
        rpr = res.get("repair", {})
        lines += [
            "",
            "  ── REPAIR MODULE RESULTS ───────────────────────────────────────",
            f"  Repair attempted:   {rpr.get('total_attempted', 0):,} samples",
            f"  Patches generated:  {rpr.get('patches_generated', 0):,}",
            f"  Patch success rate: {rpr.get('success_rate', 0):.1%}",
            "",
            "  ── TASK SCORE ANALYSIS ─────────────────────────────────────────",
        ]
        for task, stats in res.get("task_score_analysis", {}).items():
            lines.append(
                f"  {task:15s} mean={stats['mean']:.4f}  "
                f"max={stats['max']:.4f}  nonzero={stats['nonzero']:,}"
            )

        lines += [
            "",
            "  ── STRATEGY DISTRIBUTION ───────────────────────────────────────",
        ]
        for strat, cnt in res.get("strategy_distribution", {}).items():
            lines.append(f"    {strat:<22s} {cnt:,}")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Evaluation report → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MTD+ALC+Diagnosis+Repair framework"
    )
    parser.add_argument(
        "--bigvul",  default=str(ROOT / "bigvul_results.csv"),
        help="Path to bigvul_results.csv"
    )
    parser.add_argument(
        "--megavul", default=str(ROOT / "megavul_results.csv"),
        help="Path to megavul_results.csv"
    )
    parser.add_argument(
        "--dataset", default="both",
        choices=["bigvul", "megavul", "both"],
        help="Which dataset(s) to evaluate"
    )
    parser.add_argument(
        "--out", default=str(ROOT),
        help="Output directory for report files"
    )
    args    = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.dataset in ("bigvul", "both"):
        bv_rows = load_csv(Path(args.bigvul))
        if bv_rows:
            all_results["BigVul"] = evaluate_dataset(bv_rows, "BigVul")

    if args.dataset in ("megavul", "both"):
        mv_rows = load_csv(Path(args.megavul))
        if mv_rows:
            all_results["MegaVul"] = evaluate_dataset(mv_rows, "MegaVul")

    if not all_results:
        log.error(
            "No results found. Run the pipeline first:\n"
            "  python run_bigvul.py\n"
            "  python run_megavul.py"
        )
        sys.exit(1)

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = out_dir / "evaluation_report.txt"
    write_report(all_results, report_path)

    # ── Write JSON ────────────────────────────────────────────────────────────
    json_path = out_dir / "evaluation_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info(f"JSON results → {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    for ds, res in all_results.items():
        if not res:
            continue
        if res.get("labelled") is not False and "overall_metrics" in res:
            om  = res["overall_metrics"]
            auc = om.get("auc_roc_full") or om.get("auc_roc", 0)
            rv  = res.get("alc_routing_validity", {})
            print(
                f"  {ds:<10}  F1={om['f1']:.4f}  Acc={om['accuracy']:.4f}  "
                f"AUC={auc:.4f}  MCC={om['mcc']:.4f}"
            )
            if rv:
                print(
                    f"  {'':10}  ALC Delta_ALC={rv['delta_alc_pp']:+.2f} pp  "
                    f"P(Untrust|Vuln)={rv['p_untrust_given_vuln']*100:.1f}%  "
                    f"P(Untrust|NonVuln)={rv['p_untrust_given_nv']*100:.1f}%  "
                    f"{'VALID ✓' if rv['alc_valid'] else 'INVALID ✗'}"
                )
        else:
            rv   = res.get("alc_routing_validity", {})
            diag = res.get("diagnosis", {})
            rpr  = res.get("repair",    {})
            print(
                f"  {ds:<10}  [pipeline-only]  "
                f"Diagnosis={diag.get('total_run',0):,}  "
                f"Repair={rpr.get('patches_generated',0):,} patches"
            )
            if rv:
                print(
                    f"  {'':10}  ALC Delta_ALC={rv['delta_alc_pp']:+.2f} pp  "
                    f"{'VALID ✓' if rv['alc_valid'] else 'INVALID ✗'}"
                )

    print(f"\n  Reports written to: {out_dir}")
    print(f"    evaluation_report.txt")
    print(f"    evaluation_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()