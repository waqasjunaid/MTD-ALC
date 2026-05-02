# # =============================================================================
# # preprocessing/preprocess.py
# #
# # Preprocessing Pipeline Orchestrator
# #
# # Ties together all three preprocessing steps:
# #   1. Function Extraction   (function_extractor.py)
# #   2. Source File Generation (source_file_generator.py)
# #   3. Suspicious Line Mapping (suspicious_line_mapper.py)
# #
# # Reads from:
# #   data/bigvul_normalized.jsonl   (produced by prepare_bigvul.py)
# #   /mnt/data/junaid/megavul_simple_c.json
# #
# # Writes to:
# #   outputs/bigvul/sources/        .c source files
# #   outputs/megavul/sources/       .c source files
# #   data/bigvul_preprocessed.jsonl
# #   data/megavul_preprocessed.jsonl
# #
# # Each line in the output JSONL is a self-contained sample:
# #   {
# #       "id":                      str,
# #       "label":                   int,
# #       "func":                    { ...function dict... },
# #       "source_file":             str,   # path to generated .c file
# #       "line_map":                { ...LineMap dict... },
# #       "suspicious_line_numbers": [int, ...],   # ready for MTD
# #   }
# # =============================================================================
#
# import json
# import logging
# import sys
# from pathlib import Path
# from typing import Optional
#
# from function_extractor   import extract_from_dataset_row
# from source_file_generator import generate
# from suspicious_line_mapper import map_suspicious_lines, summarise
#
# # ---------------------------------------------------------------------------
# # Logging
# # ---------------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("preprocess.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# log = logging.getLogger(__name__)
#
# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------
# ROOT     = Path(__file__).resolve().parent.parent
# DATA_DIR = ROOT / "data"
# OUT_ROOT = ROOT / "outputs"
#
# BIGVUL_IN  = DATA_DIR / "bigvul_normalized.jsonl"
# MEGAVUL_IN = Path("/mnt/data/junaid/megavul_simple_c.json")
#
# BIGVUL_OUT  = DATA_DIR / "bigvul_preprocessed.jsonl"
# MEGAVUL_OUT = DATA_DIR / "megavul_preprocessed.jsonl"
#
# # ---------------------------------------------------------------------------
# # Dataset processors
# # ---------------------------------------------------------------------------
#
# def preprocess_bigvul(max_samples: Optional[int] = None):
#     """
#     Preprocess the BigVul normalized JSONL.
#     Uses GROUND_TRUTH strategy because BigVul has flaw_line_index labels.
#     """
#     if not BIGVUL_IN.exists():
#         log.error(f"BigVul input not found: {BIGVUL_IN} — run prepare_bigvul.py first")
#         return
#
#     log.info("=" * 60)
#     log.info("PREPROCESSING BigVul")
#     log.info("=" * 60)
#
#     processed = skipped = errors = 0
#
#     with open(BIGVUL_IN, encoding="utf-8") as in_fh, \
#          open(BIGVUL_OUT, "w", encoding="utf-8") as out_fh:
#
#         for raw_line in in_fh:
#             if max_samples is not None and processed >= max_samples:
#                 break
#
#             row = json.loads(raw_line)
#
#             flaw_lines = row.get("flaw_lines", [])
#             # ground_truth when the row has confirmed flaw lines (vulnerable),
#             # heuristic when flaw_lines is empty (non-vulnerable rows).
#             # Both cases are valid — non-vulnerable rows intentionally have
#             # 0 suspicious lines and must still be passed to the MTD.
#             strategy = "ground_truth" if flaw_lines else "heuristic"
#
#             try:
#                 result = _process_row(
#                     row         = row,
#                     dataset     = "bigvul",
#                     flaw_lines  = flaw_lines,
#                     strategy    = strategy,
#                 )
#             except Exception as e:
#                 log.error(f"[BigVul id={row.get('id')}] Unexpected error: {e}")
#                 errors += 1
#                 continue
#
#             if result is None:
#                 skipped += 1
#                 continue
#
#             out_fh.write(json.dumps(result) + "\n")
#             out_fh.flush()
#             processed += 1
#
#             log.info(f"[BigVul {processed}] {summarise(result['line_map'])}")
#
#     log.info(
#         f"BigVul done — processed={processed}, skipped={skipped}, errors={errors}"
#     )
#     log.info(f"Output → {BIGVUL_OUT}")
#
#
# def preprocess_megavul(max_samples: Optional[int] = None):
#     """
#     Preprocess the MegaVul JSON.
#     Uses HEURISTIC strategy because MegaVul has no flaw_line_index labels.
#     Falls back to ALL_LINES only for functions where heuristics find nothing.
#     """
#     if not MEGAVUL_IN.exists():
#         log.error(f"MegaVul input not found: {MEGAVUL_IN}")
#         return
#
#     log.info("=" * 60)
#     log.info("PREPROCESSING MegaVul")
#     log.info("=" * 60)
#
#     with open(MEGAVUL_IN, encoding="utf-8") as f:
#         dataset = json.load(f)
#
#     log.info(f"MegaVul total samples: {len(dataset)}")
#
#     processed = skipped = errors = 0
#
#     with open(MEGAVUL_OUT, "w", encoding="utf-8") as out_fh:
#
#         for idx, row in enumerate(dataset):
#             if max_samples is not None and processed >= max_samples:
#                 break
#
#             # Normalise MegaVul row to the same shape as BigVul preprocessed rows.
#             # Code is stored under "code" — extract_from_dataset_row accepts this key.
#             # MegaVul has no line-level ground-truth so flaw_lines is always empty.
#             func_code = (row.get("func_before") or row.get("code") or "").strip()
#             if not func_code:
#                 skipped += 1
#                 continue
#
#             normalised = {
#                 "id":         f"megavul_{idx}",
#                 "label":      int(row.get("target", -1)),
#                 "code":       func_code,
#                 "flaw_lines": [],   # MegaVul has no line-level labels
#             }
#
#             try:
#                 result = _process_row(
#                     row        = normalised,
#                     dataset    = "megavul",
#                     flaw_lines = [],
#                     strategy   = "heuristic",   # primary strategy
#                 )
#             except Exception as e:
#                 log.error(f"[MegaVul idx={idx}] Unexpected error: {e}")
#                 errors += 1
#                 continue
#
#             if result is None:
#                 skipped += 1
#                 continue
#
#             # Degrade to all_lines if heuristics found nothing
#             if not result["suspicious_line_numbers"]:
#                 log.warning(
#                     f"[megavul_{idx}] Heuristics found 0 suspicious lines — "
#                     f"falling back to all_lines"
#                 )
#                 result["line_map"] = map_suspicious_lines(
#                     result["func"], flaw_lines=None, strategy="all_lines"
#                 )
#                 result["suspicious_line_numbers"] = (
#                     result["line_map"]["suspicious_line_numbers"]
#                 )
#
#             out_fh.write(json.dumps(result) + "\n")
#             out_fh.flush()
#             processed += 1
#
#             log.info(f"[MegaVul {processed}] {summarise(result['line_map'])}")
#
#     log.info(
#         f"MegaVul done — processed={processed}, skipped={skipped}, errors={errors}"
#     )
#     log.info(f"Output → {MEGAVUL_OUT}")
#
#
# # ---------------------------------------------------------------------------
# # Shared row processor
# # ---------------------------------------------------------------------------
#
# def _process_row(
#     row: dict,
#     dataset: str,
#     flaw_lines: list,
#     strategy: str,
# ) -> Optional[dict]:
#     """
#     Run all three preprocessing steps for one dataset row.
#
#     Returns:
#         Combined result dict, or None if the row should be skipped.
#     """
#     # --- Step 1: Function extraction ---
#     func = extract_from_dataset_row(row, row_index=row.get("id", 0))
#     if func is None:
#         log.warning(f"[{dataset} id={row.get('id')}] No function extracted — skipping")
#         return None
#
#     # --- Step 2: Source file generation ---
#     src_path = generate(
#         func      = func,
#         out_dir   = OUT_ROOT,
#         dataset   = dataset,
#         add_main_stub = False,
#     )
#
#     # Update func dict with the actual written path
#     func["source_file"] = str(src_path)
#
#     # --- Step 3: Suspicious line mapping ---
#     line_map = map_suspicious_lines(func, flaw_lines, strategy=strategy)
#
#     return {
#         "id":                      row.get("id", func["id"]),
#         "label":                   row.get("label", -1),
#         "func":                    func,
#         "source_file":             str(src_path),
#         "line_map":                line_map,
#         "suspicious_line_numbers": line_map["suspicious_line_numbers"],
#     }
#
#
# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
#     parser.add_argument(
#         "--dataset",
#         choices=["bigvul", "megavul", "all"],
#         default="all",
#         help="Which dataset to preprocess (default: all)",
#     )
#     parser.add_argument(
#         "--max-samples",
#         type=int,
#         default=None,
#         help="Limit number of samples per dataset (default: all)",
#     )
#     args = parser.parse_args()
#
#     if args.dataset in ("bigvul", "all"):
#         preprocess_bigvul(max_samples=args.max_samples)
#
#     if args.dataset in ("megavul", "all"):
#         preprocess_megavul(max_samples=args.max_samples)
#
#     log.info("=== Preprocessing complete ===")
#
#
#

# =============================================================================
# preprocessing/preprocess.py
#
# Preprocessing Pipeline Orchestrator
#
# Ties together all three preprocessing steps:
#   1. Function Extraction   (function_extractor.py)
#   2. Source File Generation (source_file_generator.py)
#   3. Suspicious Line Mapping (suspicious_line_mapper.py)
#
# Reads from:
#   data/bigvul_normalized.jsonl   (produced by prepare_bigvul.py)
#   /mnt/data/junaid/megavul_simple_c.json
#
# Writes to:
#   outputs/bigvul/sources/        .c source files
#   outputs/megavul/sources/       .c source files
#   data/bigvul_preprocessed.jsonl
#   data/megavul_preprocessed.jsonl
#
# Each line in the output JSONL is a self-contained sample:
#   {
#       "id":                      str,
#       "label":                   int,
#       "func":                    { ...function dict... },
#       "source_file":             str,   # path to generated .c file
#       "line_map":                { ...LineMap dict... },
#       "suspicious_line_numbers": [int, ...],   # ready for MTD
#   }
# =============================================================================

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from function_extractor   import extract_from_dataset_row
from source_file_generator import generate
from suspicious_line_mapper import map_suspicious_lines, summarise

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_ROOT = ROOT / "outputs"

BIGVUL_IN  = DATA_DIR / "bigvul_normalized.jsonl"
MEGAVUL_IN = Path("/mnt/data/junaid/megavul_simple_c.json")

BIGVUL_OUT  = DATA_DIR / "bigvul_preprocessed.jsonl"
MEGAVUL_OUT = DATA_DIR / "megavul_preprocessed.jsonl"

# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def preprocess_bigvul(max_samples: Optional[int] = None):
    """
    Preprocess the BigVul normalized JSONL.
    Uses GROUND_TRUTH strategy because BigVul has flaw_line_index labels.
    """
    if not BIGVUL_IN.exists():
        log.error(f"BigVul input not found: {BIGVUL_IN} — run prepare_bigvul.py first")
        return

    log.info("=" * 60)
    log.info("PREPROCESSING BigVul")
    log.info("=" * 60)

    processed = skipped = errors = 0

    with open(BIGVUL_IN, encoding="utf-8") as in_fh, \
         open(BIGVUL_OUT, "w", encoding="utf-8") as out_fh:

        for raw_line in in_fh:
            if max_samples is not None and processed >= max_samples:
                break

            row = json.loads(raw_line)

            flaw_lines = row.get("flaw_lines", [])
            # ground_truth when the row has confirmed flaw lines (vulnerable),
            # heuristic when flaw_lines is empty (non-vulnerable rows).
            # Both cases are valid — non-vulnerable rows intentionally have
            # 0 suspicious lines and must still be passed to the MTD.
            strategy = "ground_truth" if flaw_lines else "heuristic"

            try:
                result = _process_row(
                    row         = row,
                    dataset     = "bigvul",
                    flaw_lines  = flaw_lines,
                    strategy    = strategy,
                )
            except Exception as e:
                log.error(f"[BigVul id={row.get('id')}] Unexpected error: {e}")
                errors += 1
                continue

            if result is None:
                skipped += 1
                continue

            out_fh.write(json.dumps(result) + "\n")
            out_fh.flush()
            processed += 1

            log.info(f"[BigVul {processed}] {summarise(result['line_map'])}")

    log.info(
        f"BigVul done — processed={processed}, skipped={skipped}, errors={errors}"
    )
    log.info(f"Output → {BIGVUL_OUT}")


def preprocess_megavul(max_samples: Optional[int] = None):
    """
    Preprocess MegaVul sequentially — reads entry 0, 1, 2, 3... in order.

    Each MegaVul JSON entry contains TWO functions:
        func_before  — vulnerable version before the patch  (label=1)
        func         — fixed/clean version after the patch  (label=0)

    Both are extracted from each entry in sequence:
        entry 0 → megavul_0_vul   (func_before, label=1)   sample 1
        entry 0 → megavul_0_clean (func,         label=0)   sample 2
        entry 1 → megavul_1_vul   (func_before, label=1)   sample 3
        entry 1 → megavul_1_clean (func,         label=0)   sample 4
        ...

    This gives a natural 50/50 label split in sequential order —
    no sorting, no balancing logic needed.
    Stops when max_samples total samples have been written.
    """
    if not MEGAVUL_IN.exists():
        log.error(f"MegaVul input not found: {MEGAVUL_IN}")
        return

    log.info("=" * 60)
    log.info("PREPROCESSING MegaVul  (sequential, func_before + func per entry)")
    log.info("=" * 60)

    with open(MEGAVUL_IN, encoding="utf-8") as f:
        dataset = json.load(f)

    log.info(f"MegaVul total entries in JSON: {len(dataset)}")

    processed = skipped = errors = 0

    with open(MEGAVUL_OUT, "w", encoding="utf-8") as out_fh:

        for idx, row in enumerate(dataset):
            if max_samples is not None and processed >= max_samples:
                break

            # Each entry yields two candidates in order:
            # (1) func_before = vulnerable version (label=1)
            # (2) func        = fixed/clean version (label=0)
            func_before = (row.get("func_before") or "").strip()
            func_fixed  = (row.get("func") or row.get("func_after") or "").strip()

            candidates = []
            if func_before:
                candidates.append((f"megavul_{idx}_vul",   1, func_before))
            if func_fixed and func_fixed != func_before:
                candidates.append((f"megavul_{idx}_clean", 0, func_fixed))

            if not candidates:
                log.warning(f"[MegaVul idx={idx}] No usable function code — skipping")
                skipped += 1
                continue

            for (sample_id, label, func_code) in candidates:
                if max_samples is not None and processed >= max_samples:
                    break

                normalised = {
                    "id":         sample_id,
                    "label":      label,
                    "code":       func_code,
                    "flaw_lines": [],
                }

                try:
                    result = _process_row(
                        row        = normalised,
                        dataset    = "megavul",
                        flaw_lines = [],
                        strategy   = "heuristic",
                    )
                except Exception as e:
                    log.error(f"[{sample_id}] Unexpected error: {e}")
                    errors += 1
                    continue

                if result is None:
                    skipped += 1
                    continue

                # Degrade to all_lines if heuristics find nothing
                if not result["suspicious_line_numbers"]:
                    log.warning(
                        f"[{sample_id}] 0 suspicious lines — falling back to all_lines"
                    )
                    result["line_map"] = map_suspicious_lines(
                        result["func"], flaw_lines=None, strategy="all_lines"
                    )
                    result["suspicious_line_numbers"] = (
                        result["line_map"]["suspicious_line_numbers"]
                    )

                out_fh.write(json.dumps(result) + "\n")
                out_fh.flush()
                processed += 1

                log.info(
                    f"[MegaVul {processed}] id={sample_id}  label={label}  "
                    f"{summarise(result['line_map'])}"
                )

    log.info(
        f"MegaVul done — processed={processed}, "
        f"skipped={skipped}, errors={errors}"
    )
    log.info(f"Output → {MEGAVUL_OUT}")


# ---------------------------------------------------------------------------
# Shared row processor
# ---------------------------------------------------------------------------

def _process_row(
    row: dict,
    dataset: str,
    flaw_lines: list,
    strategy: str,
) -> Optional[dict]:
    """
    Run all three preprocessing steps for one dataset row.

    Returns:
        Combined result dict, or None if the row should be skipped.
    """
    # --- Step 1: Function extraction ---
    func = extract_from_dataset_row(row, row_index=row.get("id", 0))
    if func is None:
        log.warning(f"[{dataset} id={row.get('id')}] No function extracted — skipping")
        return None

    # --- Step 2: Source file generation ---
    src_path = generate(
        func      = func,
        out_dir   = OUT_ROOT,
        dataset   = dataset,
        add_main_stub = False,
    )

    # Update func dict with the actual written path
    func["source_file"] = str(src_path)

    # --- Step 3: Suspicious line mapping ---
    line_map = map_suspicious_lines(func, flaw_lines, strategy=strategy)

    return {
        "id":                      row.get("id", func["id"]),
        "label":                   row.get("label", -1),
        "func":                    func,
        "source_file":             str(src_path),
        "line_map":                line_map,
        "suspicious_line_numbers": line_map["suspicious_line_numbers"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--dataset",
        choices=["bigvul", "megavul", "all"],
        default="all",
        help="Which dataset to preprocess (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per dataset (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("bigvul", "all"):
        preprocess_bigvul(max_samples=args.max_samples)

    if args.dataset in ("megavul", "all"):
        preprocess_megavul(max_samples=args.max_samples)

    log.info("=== Preprocessing complete ===")


