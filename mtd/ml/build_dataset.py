# =============================================================================
# mtd/ml/build_dataset.py
#
# ML Dataset Builder
#
# Reads bigvul_preprocessed.jsonl and megavul_preprocessed.jsonl, runs all
# four MTD tasks on every sample, extracts the 38-feature vector, and saves
# labelled/unlabelled feature files for training and scaler fitting.
#
# Why MegaVul is handled differently from BigVul:
#   BigVul  — has ground-truth labels (0/1).  Used for supervised training.
#   MegaVul — label=-1 (unknown).  Cannot train on it directly.
#             Used only to fit the feature scaler so it covers both dataset
#             distributions, making inference fair for both datasets.
#
# Output files:
#   datasets/bigvul_features.jsonl    — labelled   (label 0 or 1)
#   datasets/megavul_features.jsonl   — unlabelled (label -1)
#   datasets/combined_features.jsonl  — all rows from both files (for scaler)
#
# Usage:
#   python mtd/ml/build_dataset.py --dataset all --max-samples 1000
#   python mtd/ml/build_dataset.py --dataset bigvul
#   python mtd/ml/build_dataset.py --dataset megavul
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "mtd"))
sys.path.insert(0, str(ROOT / "mtd" / "ml"))

import task1_vulnerability_classification as task1
import task2_line_localization            as task2
import task3_syntax_risk_prediction       as task3
import task4_dependency_propagation_risk  as task4
from feature_extractor import extract, FEATURE_NAMES, FEATURE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"


def build(dataset_name: str, max_samples: int = None) -> int:
    """
    Extract feature vectors for one dataset and write to JSONL.

    BigVul rows with label 0/1 are written with their label.
    MegaVul rows are written with label -1 (unlabelled).
    Both are written — training code decides what to use for supervision.
    """
    input_path  = DATA_DIR / f"{dataset_name}_preprocessed.jsonl"
    output_path = DATASETS_DIR / f"{dataset_name}_features.jsonl"

    if not input_path.exists():
        log.error(
            f"Preprocessed file not found: {input_path}\n"
            f"Run first:  python preprocessing/preprocess.py --dataset {dataset_name}"
        )
        return 0

    log.info(f"[{dataset_name}] Building features from {input_path}")

    processed = skipped = errors = 0
    label_counts = Counter()

    with open(input_path, encoding="utf-8") as in_fh, \
         open(output_path, "w", encoding="utf-8") as out_fh:

        for line_num, raw_line in enumerate(in_fh, start=1):

            if max_samples is not None and processed >= max_samples:
                break

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_num}: JSON error — {e}")
                skipped += 1
                continue

            if not row.get("source_file") or not row.get("func") \
                    or not row.get("line_map"):
                log.warning(f"Line {line_num}: missing required fields — skipping")
                skipped += 1
                continue

            if not Path(row["source_file"]).exists():
                log.warning(f"Line {line_num}: source file missing on disk — skipping")
                skipped += 1
                continue

            sample_id        = str(row["id"])
            label            = int(row.get("label", -1))
            source_file      = row["source_file"]
            suspicious_lines = row.get("suspicious_line_numbers", [])
            func             = row["func"]
            line_map         = row["line_map"]
            strategy         = line_map.get("strategy", "unknown")

            try:
                r1 = task1.run(source_file, suspicious_lines, func, line_map)
                r2 = task2.run(source_file, suspicious_lines, func, line_map)
                r3 = task3.run(source_file, suspicious_lines, func, line_map)
                r4 = task4.run(source_file, suspicious_lines, func, line_map)
                features = extract(r1, r2, r3, r4, func, line_map, suspicious_lines)
            except Exception as e:
                log.error(f"Sample {sample_id}: feature extraction failed — {e}")
                errors += 1
                continue

            out_fh.write(json.dumps({
                "sample_id":     sample_id,
                "dataset":       dataset_name,
                "label":         label,
                "strategy":      strategy,
                "features":      features,
                "feature_names": FEATURE_NAMES,
            }) + "\n")
            out_fh.flush()

            processed += 1
            label_counts[label] += 1

            if processed % 500 == 0 or processed <= 5:
                log.info(
                    f"  [{dataset_name}] {processed} samples  "
                    f"labels={dict(label_counts)}  errors={errors}"
                )

    log.info(
        f"[{dataset_name}] DONE — processed={processed}  "
        f"skipped={skipped}  errors={errors}  "
        f"label_dist={dict(label_counts)}"
    )

    # Warn if no vulnerable (label=1) samples were found
    if dataset_name == "bigvul" and label_counts.get(1, 0) == 0:
        log.warning(
            f"[bigvul] WARNING: 0 VULNERABLE samples (label=1) in the output.\n"
            f"The first {processed} BigVul samples are all non-vulnerable.\n"
            f"Increase --max-samples to include vulnerable samples:\n"
            f"  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
        )

    if dataset_name == "megavul" and label_counts.get(-1, 0) > 0:
        log.info(
            f"[megavul] NOTE: all {label_counts[-1]} samples have label=-1 "
            f"(no ground truth). Used for scaler fitting only, not for training."
        )
    return processed


def combine() -> int:
    """
    Merge bigvul and megavul feature files into combined_features.jsonl.
    Used to fit a scaler that covers both dataset distributions.
    """
    combined_path = DATASETS_DIR / "combined_features.jsonl"
    total = 0

    with open(combined_path, "w", encoding="utf-8") as out_fh:
        for name in ("bigvul", "megavul"):
            p = DATASETS_DIR / f"{name}_features.jsonl"
            if not p.exists():
                log.warning(f"[combine] {name}_features.jsonl not found — skipping")
                continue
            count = 0
            with open(p, encoding="utf-8") as in_fh:
                for line in in_fh:
                    line = line.strip()
                    if line:
                        out_fh.write(line + "\n")
                        count += 1
                        total += 1
            log.info(f"[combine] Added {count} samples from {name}")

    log.info(f"[combine] Combined dataset: {total} total → {combined_path}")
    return total


def print_summary():
    """Print a summary of what is available in each feature file."""
    log.info("=== Dataset Summary ===")
    for name in ("bigvul", "megavul", "combined"):
        p = DATASETS_DIR / f"{name}_features.jsonl"
        if not p.exists():
            log.info(f"  {name:12s}: NOT BUILT")
            continue
        counts = Counter()
        total  = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                counts[rec.get("label", -1)] += 1
                total += 1
        labelled   = counts.get(0, 0) + counts.get(1, 0)
        unlabelled = counts.get(-1, 0)
        log.info(
            f"  {name:12s}: total={total}  "
            f"labelled={labelled} (0:{counts.get(0,0)} 1:{counts.get(1,0)})  "
            f"unlabelled={unlabelled}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ML feature dataset")
    parser.add_argument(
        "--dataset", choices=["bigvul", "megavul", "all"], default="all",
        help="Which dataset to build features for (default: all)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per dataset (default: all)"
    )
    args = parser.parse_args()

    if args.dataset in ("bigvul", "all"):
        build("bigvul", args.max_samples)

    if args.dataset in ("megavul", "all"):
        build("megavul", args.max_samples)

    combine()
    print_summary()
    log.info("Dataset build complete.")