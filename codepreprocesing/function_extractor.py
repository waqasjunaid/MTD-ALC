# =============================================================================
# preprocessing/function_extractor.py
#
# Function Extraction
#
# Extracts individual C/C++ functions from raw source code or dataset rows.
# Handles:
#   - Single-function strings (BigVul / MegaVul rows already contain one func)
#   - Multi-function source files (raw .c / .cpp files)
#   - Malformed / incomplete functions (brace mismatch, empty bodies)
#
# Output per function:
#   {
#       "id":           str,         # unique identifier
#       "name":         str,         # function name
#       "code":         str,         # full function source text
#       "start_line":   int,         # 1-based line in the original file
#       "end_line":     int,
#       "param_count":  int,         # number of formal parameters
#       "line_count":   int,         # total lines in function body
#       "source_file":  str | None,  # path to origin file if known
#   }
# =============================================================================

import re
import logging
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a C/C++ function signature:
#   return_type  function_name  (  params  )
# Deliberately loose — handles pointer returns, qualifiers, macros, etc.
_FUNC_SIG = re.compile(
    r"""
    (?:^|\n)                           # start of string or new line
    (?:                                # optional qualifiers / return type
        (?:static|inline|extern|const|volatile|unsigned|signed|struct|enum|
           __attribute__\s*\(.*?\))\s+
    )*
    (?:[\w\*\s]+?)                     # return type (greedy but minimal)
    \s+
    (?P<name>[\w]+)                    # function name  ← captured
    \s*
    \(                                 # opening paren of param list
    (?P<params>[^)]*)                  # parameters     ← captured
    \)
    \s*
    \{                                 # opening brace
    """,
    re.VERBOSE | re.MULTILINE,
)

# Detects lines that are clearly NOT function definitions
_SKIP_LINE = re.compile(
    r"^\s*("
    r"#|"                # preprocessor directives
    r"//|"               # single-line comments
    r"/\*|"              # block comment start
    r"\*|"               # block comment continuation
    r"typedef\s|"        # typedefs
    r"struct\s+\w+\s*\{|"  # struct definitions
    r"enum\s+\w+\s*\{"   # enum definitions
    r")"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_string(
    code: str,
    source_id: str = "inline",
    source_file: Optional[str] = None,
) -> list[dict]:
    """
    Extract all functions from a raw code string.

    Args:
        code:        Raw C/C++ source text.
        source_id:   Prefix used to build unique function IDs.
        source_file: Optional path label stored in the output dict.

    Returns:
        List of function dicts (see module docstring for schema).
    """
    if not code or not code.strip():
        return []

    # Strip block comments so brace counting isn't confused by /* { */
    clean = _strip_block_comments(code)

    functions = []
    lines = code.splitlines()          # keep originals for output
    clean_lines = clean.splitlines()

    i = 0
    func_index = 0

    while i < len(clean_lines):
        line = clean_lines[i]

        # Quick skip for lines that definitely aren't function starts
        if _SKIP_LINE.match(line):
            i += 1
            continue

        # Try to find a function signature starting at line i
        # We search a small window (up to 6 lines) to handle multi-line sigs
        window = "\n".join(clean_lines[i : i + 6])
        m = _FUNC_SIG.search(window)

        if not m:
            i += 1
            continue

        func_name = m.group("name")
        raw_params = m.group("params").strip()

        # Locate the opening brace in the window, then walk to closing brace
        # We need the absolute line of the opening brace
        brace_offset = window.find("{", m.start())
        if brace_offset == -1:
            i += 1
            continue

        brace_line_offset = window[: brace_offset].count("\n")
        open_brace_line = i + brace_line_offset   # 0-based

        # Walk forward counting braces to find the closing brace
        end_line = _find_closing_brace(clean_lines, open_brace_line)
        if end_line is None:
            log.debug(f"[{source_id}] Unmatched brace starting at line {i+1} — skipping")
            i += 1
            continue

        # Extract original (non-stripped) lines for the function body
        func_lines = lines[i : end_line + 1]
        func_code = "\n".join(func_lines).strip()

        if not _is_valid_function(func_code):
            i = end_line + 1
            continue

        param_count = _count_params(raw_params)

        func_dict = {
            "id":          f"{source_id}_func{func_index}",
            "name":        func_name,
            "code":        func_code,
            "start_line":  i + 1,          # 1-based
            "end_line":    end_line + 1,    # 1-based
            "param_count": param_count,
            "line_count":  end_line - i + 1,
            "source_file": source_file,
        }

        functions.append(func_dict)
        log.debug(
            f"[{source_id}] Extracted '{func_name}' "
            f"lines {i+1}–{end_line+1} ({func_dict['line_count']} lines)"
        )

        func_index += 1
        i = end_line + 1   # jump past this function

    return functions


def extract_from_file(filepath: Union[str, Path]) -> list[dict]:
    """
    Read a .c / .cpp file from disk and extract all functions.

    Args:
        filepath: Path to source file.

    Returns:
        List of function dicts.
    """
    p = Path(filepath)
    if not p.exists():
        log.error(f"File not found: {p}")
        return []

    try:
        code = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.error(f"Cannot read {p}: {e}")
        return []

    return extract_from_string(code, source_id=p.stem, source_file=str(p))


def extract_from_dataset_row(row: dict, row_index: int = 0) -> Optional[dict]:
    """
    Extract a single function from a BigVul / MegaVul dataset row.

    Accepts rows from two sources:
      - prepare_bigvul.py output  -> code stored under key "code"
      - raw MegaVul JSON          -> code stored under key "func_before"
    Both are tried in order so this works for either dataset without the
    caller needing to normalise the key first.

    Because dataset rows already contain exactly one function, start_line is
    always forced to 1 (line numbers are function-relative, not file-relative).
    flaw_lines from prepare_bigvul.py are already function-relative 1-based
    numbers, so no offset conversion is needed in suspicious_line_mapper.

    Args:
        row:       Dict with either a "code" or "func_before" key.
        row_index: Used to build the unique ID when row has no "id" field.

    Returns:
        A single function dict, or None if the row has no usable code.
    """
    # Accept either key: "code" (prepare_bigvul output) or "func_before" (raw dataset)
    code = (row.get("code") or row.get("func_before") or "").strip()
    if not code:
        return None

    sample_id = str(row.get("id", row_index))
    functions = extract_from_string(code, source_id=sample_id)

    if not functions:
        # Fallback: treat whole string as one function if signature parsing fails
        log.warning(
            f"[{sample_id}] Signature parser found no functions — using raw code as-is"
        )
        return {
            "id":          f"{sample_id}_func0",
            "name":        f"unknown_{sample_id}",
            "code":        code,
            "start_line":  1,   # always 1 — dataset rows are single-function strings
            "end_line":    len(code.splitlines()),
            "param_count": 0,
            "line_count":  len(code.splitlines()),
            "source_file": None,
        }

    # Return the first (and usually only) extracted function.
    # Force start_line=1 because dataset rows are single-function strings —
    # flaw_lines from prepare_bigvul.py index from line 1 of the function string.
    func = functions[0]
    func["start_line"] = 1
    return func


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_block_comments(code: str) -> str:
    """Remove /* ... */ block comments, preserving line count."""
    result = []
    i = 0
    while i < len(code):
        if code[i : i + 2] == "/*":
            end = code.find("*/", i + 2)
            if end == -1:
                break
            # Replace comment content with spaces, keep newlines intact
            segment = code[i : end + 2]
            result.append(re.sub(r"[^\n]", " ", segment))
            i = end + 2
        else:
            result.append(code[i])
            i += 1
    return "".join(result)


def _find_closing_brace(lines: list[str], open_brace_line: int) -> Optional[int]:
    """
    Given the 0-based index of the line containing the opening '{',
    return the 0-based index of the line containing the matching '}'.
    Returns None if no match is found.
    """
    depth = 0
    for i in range(open_brace_line, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
    return None


def _count_params(params_str: str) -> int:
    """Count formal parameters from a raw parameter string."""
    if not params_str or params_str.strip() in ("", "void"):
        return 0
    # Split on commas, but ignore commas inside nested parens (function pointers)
    depth = 0
    count = 1
    for ch in params_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def _is_valid_function(code: str) -> bool:
    """Basic sanity checks — reject obviously bad extractions."""
    if len(code.strip()) < 10:
        return False
    if code.count("{") != code.count("}"):
        return False
    # Must have at least one statement (a semicolon or a closing brace line)
    if ";" not in code and code.count("}") < 2:
        return False
    return True


# ---------------------------------------------------------------------------
# CLI entry point (for quick testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python function_extractor.py <source_file.c>")
        sys.exit(1)

    funcs = extract_from_file(sys.argv[1])
    print(json.dumps(funcs, indent=2))
    print(f"\nExtracted {len(funcs)} function(s)")

