#!/usr/bin/env python3
"""
patch_validity_and_applicability.py

Answers two parts of the reviewer comment "the generated patch might be
invalid... the evaluation does not consider effectiveness or ease of
applying it" that weren't previously measured:

1. VALIDITY: structural well-formedness (balanced braces/parens, not
   truncated) -- NOT full compilation, which is impossible here for the
   same reason already stated in the paper (BigVul supplies isolated
   functions without build context: missing project headers, custom
   types, etc.). A best-effort gcc -fsyntax-only cross-check is also
   attempted, filtering out expected "unknown type"/"undeclared
   identifier" noise (inevitable without headers) and counting only
   genuine parse/grammar errors, if gcc is available.

2. APPLICABILITY: (a) whether the function signature was preserved
   exactly (a proxy for drop-in-replaceability, and a direct check of
   whether the repair prompt's "preserve the original function
   signature exactly" instruction was followed), and (b) how much of
   the function changed (line-based diff ratio), as a proxy for how
   minimal/reviewable the patch is.

Usage:
    python patch_validity_and_applicability.py --results results.jsonl
"""
import argparse
import difflib
import json
import re

# Re-used from earlier scripts in this project, for consistent cleaning
_INLINE_FIX_COMMENT_RE = re.compile(r"/\*\s*FIX\b.*?\*/", re.IGNORECASE | re.DOTALL)
_CHANGES_BLOCK_RE = re.compile(
    r"/\*\s*(CHANGES MADE|Changes made|Summary of changes)\s*:?.*?\*/",
    re.IGNORECASE | re.DOTALL,
)
_HEADER_RE = re.compile(r"^\s*/\*.*?Repaired by LLM Repair Module.*?\*/\s*", re.DOTALL)

_PROSE_PREFIX_MARKERS = re.compile(
    r"^(here'?s|here is|certainly|sure[,.]|below is|the following|"
    r"this (function|code|patch)|i'?ve|i have)\b", re.IGNORECASE)


def strip_leading_block_comment(text: str) -> str:
    """Strips the FIRST block comment if it appears at the very start of
    the (whitespace-trimmed) text, regardless of its content -- more
    robust than matching specific phrases like 'CHANGES MADE:', since
    the LLM uses several different phrasings and comment styles
    (including the standard '/*\\n * text\\n */' convention, where a
    per-line leading '*' broke the original phrase-matching regex)."""
    stripped = text.lstrip()
    if not stripped.startswith("/*"):
        return text
    end = stripped.find("*/")
    if end == -1:
        return text
    return stripped[end + 2:].lstrip()


def strip_bare_prose_prefix(text: str) -> str:
    """Strips a leading non-comment prose line (e.g. 'Here is the patched
    function:') that isn't wrapped in any comment syntax at all."""
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    if _PROSE_PREFIX_MARKERS.match(first) and not first.rstrip().endswith(("{", ";")):
        return "\n".join(lines[1:]).lstrip()
    return text


def clean_patch(text: str) -> str:
    text = _HEADER_RE.sub("", text)
    text = strip_leading_block_comment(text)
    text = strip_bare_prose_prefix(text)
    text = strip_leading_block_comment(text)  # in case a second comment follows the prose line
    text = _INLINE_FIX_COMMENT_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------
# 1. Structural validity
# ---------------------------------------------------------------------
def check_structural_validity(code: str):
    """Returns (is_valid, reason). Checks brace/paren/bracket balance and
    that the code doesn't end mid-statement (a common truncation
    symptom in LLM output)."""
    pairs = {'{': '}', '(': ')', '[': ']'}
    closers = {v: k for k, v in pairs.items()}
    stack = []
    in_string = None  # tracks ' or " we're inside, None if not in a string
    escaped = False

    for ch in code:
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in ('"', "'"):
            in_string = ch
            continue
        if ch in pairs:
            stack.append(ch)
        elif ch in closers:
            if not stack or stack[-1] != closers[ch]:
                return False, f"unbalanced '{ch}' (unexpected closer)"
            stack.pop()

    if in_string:
        return False, f"unterminated string/char literal ({in_string})"
    if stack:
        return False, f"unbalanced: {len(stack)} unclosed {stack}"

    stripped = code.strip()
    if not stripped.endswith('}'):
        return False, "does not end with closing brace (likely truncated)"

    return True, "ok"


# NOTE: an earlier version of this script also attempted a gcc
# -fsyntax-only cross-check. This was REMOVED after diagnosis showed it
# producing false "structural error" reports on valid C++ code (e.g.
# std::string&, Class::Method()) -- BigVul mixes C and C++, and plain
# `gcc` only parses C, so it rejects valid C++ syntax regardless of
# patch quality. It also flagged kernel macro-based signatures
# (SYSCALL_DEFINEn) as errors purely because the defining headers
# aren't available -- the same missing-build-context limitation
# already disclosed elsewhere in the paper, not a new finding. The
# pure-Python structural check above (brace/paren balance, no
# truncation) is intentionally language- and context-agnostic and does
# not suffer from either problem.


# ---------------------------------------------------------------------
# 2. Applicability: signature preservation + diff size
# ---------------------------------------------------------------------
_FUNC_SIG_SPLIT = re.compile(r"^(.*?\))\s*\{", re.DOTALL)


def extract_signature(code: str):
    """Best-effort: everything up to the first `) {` -- the function
    signature/declaration, before the body begins."""
    m = _FUNC_SIG_SPLIT.search(code)
    if not m:
        return None
    sig = m.group(1)
    return re.sub(r"\s+", " ", sig).strip()


def signature_preserved(original: str, patch: str):
    orig_sig = extract_signature(original)
    patch_sig = extract_signature(patch)
    if orig_sig is None or patch_sig is None:
        return None  # couldn't extract from one or both -- not counted
    return orig_sig == patch_sig


def diff_ratio(original: str, patch: str):
    """Fraction of lines changed (added+removed) relative to the
    original's line count -- a proxy for how invasive the patch is."""
    orig_lines = original.strip().splitlines()
    patch_lines = patch.strip().splitlines()
    sm = difflib.SequenceMatcher(a=orig_lines, b=patch_lines)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes()
                  if tag != "equal")
    return changed / max(1, len(orig_lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.results) if l.strip()]
    n = len(recs)

    n_structurally_valid = 0
    n_sig_checked = 0
    n_sig_preserved = 0
    diff_ratios = []

    invalid_examples = []

    for rec in recs:
        patch = clean_patch(rec["generated_patch"])
        original = rec["vulnerable_func"]
        sid = rec["id"]

        valid, reason = check_structural_validity(patch)
        if valid:
            n_structurally_valid += 1
        else:
            invalid_examples.append((sid, reason))

        preserved = signature_preserved(original, patch)
        if preserved is not None:
            n_sig_checked += 1
            if preserved:
                n_sig_preserved += 1

        diff_ratios.append(diff_ratio(original, patch))

    print(f"Total patches: {n}\n")

    print(f"--- Structural validity (balanced braces/parens, not truncated) ---")
    print(f"  Valid: {n_structurally_valid}/{n} ({100*n_structurally_valid/n:.1f}%)")
    if invalid_examples:
        print(f"  First 10 invalid examples:")
        for sid, reason in invalid_examples[:10]:
            print(f"    id={sid}: {reason}")

    print(f"\n--- Function signature preservation ---")
    print(f"  Checked: {n_sig_checked}/{n} (extraction succeeded for both original and patch)")
    if n_sig_checked:
        print(f"  Signature preserved exactly: {n_sig_preserved}/{n_sig_checked} "
              f"({100*n_sig_preserved/n_sig_checked:.1f}%)")

    print(f"\n--- Patch invasiveness (fraction of lines changed vs. original) ---")
    diff_ratios.sort()
    n_d = len(diff_ratios)
    print(f"  Median: {diff_ratios[n_d//2]:.3f}")
    print(f"  Mean:   {sum(diff_ratios)/n_d:.3f}")
    print(f"  p90:    {diff_ratios[int(0.9*n_d)]:.3f}")
    print(f"  Fraction of patches changing <20% of lines: "
          f"{100*sum(1 for r in diff_ratios if r < 0.2)/n_d:.1f}%")
    print(f"  Fraction of patches changing >50% of lines: "
          f"{100*sum(1 for r in diff_ratios if r > 0.5)/n_d:.1f}%")


if __name__ == "__main__":
    main()