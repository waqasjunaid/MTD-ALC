#!/usr/bin/env python3
"""
regen_heuristic_preprocessed.py

Produce a copy of the BigVul preprocessed file in which EVERY sample uses the
heuristic suspicious-line mapping, by re-mapping each function with the
ground-truth flaw lines withheld (flaw_lines=[], strategy="heuristic").

Purpose: answer the reviewer request to report BigVul results separately under
heuristic mapping. The current run uses ground_truth for the 788 samples whose
CVE flaw lines align and heuristic for the rest; this regenerates a fully-heuristic
version so run_bigvul.py can produce a clean heuristic-condition results file whose
only difference from the reported run is those 788 samples.

It reuses your real mapper (codepreprocesing/suspicious_line_mapper.py), so the
heuristic mapping is identical to the one used everywhere else in the pipeline.

Self-check (default): re-maps 3 samples, prints old vs new strategy and flagged
counts, and stops. Add --full to write the whole file.

Usage:
  python regen_heuristic_preprocessed.py --root . \
      --in data/bigvul_preprocessed.jsonl
  # then, once it looks right:
  python regen_heuristic_preprocessed.py --root . \
      --in data/bigvul_preprocessed.jsonl \
      --out data/bigvul_preprocessed_heuristic.jsonl --full
"""

import argparse
import json
import sys
from pathlib import Path


def add_paths(root):
    for p in [root, root / "codepreprocesing"]:
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def load_mapper(root):
    add_paths(root)
    from suspicious_line_mapper import map_suspicious_lines
    return map_suspicious_lines


def remap_record(rec, map_fn):
    """Return a copy of the preprocessed record with a heuristic line_map."""
    func = rec.get("func", {})
    # force heuristic: withhold flaw lines, pin strategy
    new_map = map_fn(func, flaw_lines=[], strategy="heuristic")
    out = dict(rec)
    out["line_map"] = new_map
    out["suspicious_line_numbers"] = new_map.get(
        "suspicious_line_numbers",
        [e["line_no"] for e in new_map.get("suspicious_lines", [])],
    )
    return out, rec.get("line_map", {}).get("strategy"), new_map.get("strategy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--in", dest="infile", default="data/bigvul_preprocessed.jsonl")
    ap.add_argument("--out", default="data/bigvul_preprocessed_heuristic.jsonl")
    ap.add_argument("--full", action="store_true", help="write all samples (default: 3-sample self-check)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    infile = Path(args.infile)
    if not infile.exists():
        sys.exit(f"ERROR: input not found: {infile}")

    try:
        map_fn = load_mapper(root)
    except Exception as e:
        sys.exit(f"ERROR importing mapper (check --root={root}): {e}")

    lines = [l for l in open(infile, encoding="utf-8") if l.strip()]
    print(f"loaded {len(lines)} preprocessed records from {infile}")

    if not args.full:
        print("\n*** SELF-CHECK (3 samples). Add --full to write the file. ***\n")
        shown = 0
        for l in lines:
            rec = json.loads(l)
            func = rec.get("func", {})
            if "code" not in func:
                print(f"  id={rec.get('id')}: WARNING func has no 'code' key; "
                      f"keys={list(func)[:8]} -- mapper may fail")
            try:
                out, old_s, new_s = remap_record(rec, map_fn)
                old_n = len(rec.get("suspicious_line_numbers", []))
                new_n = len(out["suspicious_line_numbers"])
                print(f"  id={rec.get('id')}  strategy {old_s} -> {new_s}  "
                      f"suspicious {old_n} -> {new_n}")
            except Exception as e:
                print(f"  id={rec.get('id')}: remap FAILED: {e}")
            shown += 1
            if shown >= 3:
                break
        print("\nIf strategy shows '... -> heuristic' and remap did not fail, "
              "re-run with --full.")
        return

    n_written = 0
    strat_before = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for i, l in enumerate(lines, 1):
            rec = json.loads(l)
            s0 = rec.get("line_map", {}).get("strategy", "?")
            strat_before[s0] = strat_before.get(s0, 0) + 1
            out, _, _ = remap_record(rec, map_fn)
            f.write(json.dumps(out) + "\n")
            n_written += 1
            if i % 2000 == 0 or i == len(lines):
                print(f"  remapped {i}/{len(lines)}", flush=True)

    print(f"\nwrote {n_written} records to {args.out}")
    print(f"original strategy distribution: {strat_before}")
    print("all records in the output now use heuristic mapping.")
    print(f"\nNext: run your pipeline against {args.out} to produce a "
          f"heuristic-condition results CSV (see instructions).")


if __name__ == "__main__":
    main()
