#!/usr/bin/env python3
"""
diagnose_validity_applicability.py

Investigates whether the low signature-preservation and gcc-validity
numbers from patch_validity_and_applicability.py are real findings or
measurement artifacts, by printing concrete examples.

Usage:
    python diagnose_validity_applicability.py --results results.jsonl
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_INLINE_FIX_COMMENT_RE = re.compile(r"/\*\s*FIX\b.*?\*/", re.IGNORECASE | re.DOTALL)
_CHANGES_BLOCK_RE = re.compile(
    r"/\*\s*(CHANGES MADE|Changes made|Summary of changes)\s*:?.*?\*/",
    re.IGNORECASE | re.DOTALL,
)
_HEADER_RE = re.compile(r"^\s*/\*.*?Repaired by LLM Repair Module.*?\*/\s*", re.DOTALL)


def clean_patch(text):
    text = _HEADER_RE.sub("", text)
    text = _CHANGES_BLOCK_RE.sub("", text)
    text = _INLINE_FIX_COMMENT_RE.sub("", text)
    return text.strip()


_FUNC_SIG_SPLIT = re.compile(r"^(.*?\))\s*\{", re.DOTALL)


def extract_signature(code):
    m = _FUNC_SIG_SPLIT.search(code)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip()


GCC_STRUCTURAL_ERROR_PATTERNS = [
    r"expected .* before", r"expected .* at end of input", r"expected expression",
    r"expected identifier or", r"stray .* in program", r"unterminated comment",
    r"missing terminating",
]
GCC_STRUCTURAL_ERROR_RE = re.compile("|".join(GCC_STRUCTURAL_ERROR_PATTERNS), re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.results) if l.strip()]

    print("=" * 70)
    print("PART 1: Signature preservation -- first 5 'not preserved' examples")
    print("=" * 70)
    shown = 0
    for rec in recs:
        original = rec["vulnerable_func"]
        patch = clean_patch(rec["generated_patch"])
        orig_sig = extract_signature(original)
        patch_sig = extract_signature(patch)
        if orig_sig is None or patch_sig is None:
            continue
        if orig_sig != patch_sig:
            print(f"\nid={rec['id']}")
            print(f"  ORIGINAL sig: {orig_sig[:200]!r}")
            print(f"  PATCH    sig: {patch_sig[:200]!r}")
            shown += 1
            if shown >= 5:
                break

    print("\n" + "=" * 70)
    print("PART 2: gcc failures on structurally-valid patches -- first 3 examples")
    print("=" * 70)
    if shutil.which("gcc") is None:
        print("gcc not available, skipping")
        return

    shown = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for rec in recs:
            patch = clean_patch(rec["generated_patch"])
            path = tmpdir / f"tmp_{rec['id']}.c"
            path.write_text(patch, encoding="utf-8", errors="replace")
            proc = subprocess.run(
                ["gcc", "-fsyntax-only", "-w", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            structural_errors = [l for l in proc.stderr.splitlines()
                                 if GCC_STRUCTURAL_ERROR_RE.search(l)]
            if structural_errors:
                print(f"\nid={rec['id']}")
                print(f"  First 3 gcc lines flagged as 'structural':")
                for l in structural_errors[:3]:
                    print(f"    {l}")
                shown += 1
                if shown >= 3:
                    break


if __name__ == "__main__":
    main()
