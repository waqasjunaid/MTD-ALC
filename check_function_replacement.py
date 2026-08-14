#!/usr/bin/env python3
"""
check_function_replacement.py

Some BigVul commits don't patch a function's body in place -- they
delete/rename it and introduce a differently-named replacement. When
that happens, reference_fix contains the OLD function's signature
(as unchanged diff context) immediately followed by a NEW, differently
named function. No LLM patch that (correctly) keeps the original
function's name and signature can ever match such a reference.

This script flags reference_fix entries where a second, differently
named function-like signature appears within the first ~300 characters
-- i.e. the "replacement" pattern seen in id=11 and id=3660.

Usage:
    python check_function_replacement.py diagnosis_records.jsonl
"""

import json
import re
import sys

# Matches genuine function DEFINITIONS: name(params) immediately
# followed by '{' (a param list with no ';' or '{' inside it, e.g. not
# a call like `foo(bar());` or a control structure). This deliberately
# excludes ordinary function CALLS, which are followed by ';', ',', ')',
# etc., not directly by '{'.
FUNC_DEF_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{")
SKIP = {"if", "for", "while", "switch", "do", "catch"}


def find_func_names(text, limit_chars=400):
    names = []
    for m in FUNC_DEF_RE.finditer(text[:limit_chars]):
        name = m.group(1)
        if name not in SKIP:
            names.append(name)
    return names


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
            vuln_names = find_func_names(r.get("vulnerable_func", ""), limit_chars=200)
            fix_names = find_func_names(r.get("reference_fix", ""), limit_chars=400)

            if not vuln_names or not fix_names:
                continue

            vuln_primary = vuln_names[0]
            # does a DIFFERENT function name show up in the fix, distinct
            # from the vulnerable function's own name?
            other_names = [nm for nm in fix_names if nm != vuln_primary]
            if fix_names[0] == vuln_primary and other_names:
                flagged.append((r.get("id"), vuln_primary, other_names[:3]))

    print(f"total records: {n}")
    print(f"flagged as possible function-replacement pattern: {len(flagged)} "
          f"({100*len(flagged)/n:.1f}%)")
    if flagged:
        print("\n--- first 10 flagged ids ---")
        for id_, vname, others in flagged[:10]:
            print(f"  id={id_}  original={vname!r}  other_names_seen={others}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_function_replacement.py diagnosis_records.jsonl")
        sys.exit(1)
    main(sys.argv[1])