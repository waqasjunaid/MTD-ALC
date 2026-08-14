#!/usr/bin/env python3
"""
check_patch_prose.py

Flags generated_patch entries in results.jsonl that likely contain
leftover natural-language prose (not stripped by _strip_fences, which
only removes markdown code fences) -- e.g. "Here's the fixed function:"
or a trailing "Explanation:" paragraph. Any such text guarantees
exact-match will fail even when the code itself is a correct fix.

Usage:
    python check_patch_prose.py results.jsonl
"""

import json
import re
import sys

PROSE_MARKERS = [
    r"^here'?s\b", r"^here is\b", r"^certainly\b", r"^sure[,.]",
    r"^below is\b", r"^the following\b", r"^this (function|code|patch)\b",
    r"^i'?ve\b", r"^i have\b", r"^explanation\s*:", r"^note\s*:",
    r"^fix\s*:", r"^summary\s*:", r"```",
]
PROSE_RE = re.compile("|".join(PROSE_MARKERS), re.IGNORECASE)


def first_nonblank_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def last_nonblank_line(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def main(path):
    n = 0
    flagged = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            patch = r.get("generated_patch", "")
            head = first_nonblank_line(patch)
            tail = last_nonblank_line(patch)

            reason = None
            if PROSE_RE.search(head):
                reason = f"first line looks like prose: {head[:80]!r}"
            elif PROSE_RE.search(tail):
                reason = f"last line looks like prose: {tail[:80]!r}"

            if reason:
                flagged.append((r.get("id"), reason, patch[:150], patch[-150:]))

    print(f"total patches checked: {n}")
    print(f"flagged as likely containing prose: {len(flagged)} "
          f"({100*len(flagged)/n:.1f}%)")

    if flagged:
        print("\n--- first 5 flagged examples ---")
        for id_, reason, head, tail in flagged[:5]:
            print(f"\nid={id_}  {reason}")
            print(f"  PATCH HEAD: {head!r}")
            print(f"  PATCH TAIL: {tail!r}")
    else:
        print("\nNo obvious prose markers found -- 0% exact-match likely "
              "reflects genuine token/formatting differences rather than "
              "a text-extraction bug. Worth manually eyeballing 2-3 pairs "
              "of (generated_patch, reference_fix) directly to confirm.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_patch_prose.py results.jsonl")
        sys.exit(1)
    main(sys.argv[1])
