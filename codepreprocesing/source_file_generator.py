# =============================================================================
# preprocessing/source_file_generator.py
#
# Source File Generation
#
# Takes an extracted function dict (from function_extractor.py) and writes
# it to a self-contained, compilable .c source file on disk.
#
# Why this matters:
#   Static analysis tools (cppcheck, clang, joern, etc.) need real files —
#   they cannot operate on raw strings. This module:
#     1. Adds required standard headers based on symbols detected in the code
#     2. Wraps the function in a minimal compilation unit
#     3. Appends a stub main() only when needed for standalone compilation
#     4. Writes the file and returns the output path
#
# Output file layout:
#   <out_dir>/<dataset>/<sample_id>.c
#
# Each file is deterministic — running twice with the same input produces
# the same file (safe for caching / incremental pipelines).
# =============================================================================

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header inference table
# Symbol pattern  →  header to include
# ---------------------------------------------------------------------------
_HEADER_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(printf|fprintf|scanf|sprintf|snprintf|sscanf|puts|gets|fgets|"
                r"fopen|fclose|fread|fwrite|fseek|ftell|rewind|feof|ferror|"
                r"perror|remove|rename|tmpfile|EOF)\b"),        "<stdio.h>"),
    (re.compile(r"\b(malloc|calloc|realloc|free|exit|abort|getenv|system|"
                r"atoi|atol|atof|strtol|strtod|rand|srand|abs|labs|qsort|"
                r"bsearch|NULL)\b"),                            "<stdlib.h>"),
    (re.compile(r"\b(strlen|strcpy|strncpy|strcat|strncat|strcmp|strncmp|"
                r"strchr|strrchr|strstr|strtok|memcpy|memmove|memset|memcmp|"
                r"memchr)\b"),                                  "<string.h>"),
    (re.compile(r"\b(isalpha|isdigit|isalnum|isspace|isupper|islower|"
                r"toupper|tolower|isprint|ispunct|iscntrl)\b"), "<ctype.h>"),
    (re.compile(r"\b(sin|cos|tan|asin|acos|atan|atan2|sqrt|pow|exp|log|"
                r"log10|ceil|floor|fabs|fmod|M_PI)\b"),         "<math.h>"),
    (re.compile(r"\b(time|clock|difftime|mktime|localtime|gmtime|"
                r"strftime|asctime|ctime)\b"),                  "<time.h>"),
    (re.compile(r"\b(assert)\b"),                               "<assert.h>"),
    (re.compile(r"\b(va_list|va_start|va_end|va_arg)\b"),       "<stdarg.h>"),
    (re.compile(r"\b(int8_t|int16_t|int32_t|int64_t|"
                r"uint8_t|uint16_t|uint32_t|uint64_t|"
                r"size_t|ptrdiff_t|intptr_t|uintptr_t)\b"),     "<stdint.h>"),
    (re.compile(r"\b(bool|true|false)\b"),                      "<stdbool.h>"),
    (re.compile(r"\b(errno|ENOENT|EINVAL|EACCES|ENOMEM)\b"),    "<errno.h>"),
    (re.compile(r"\b(open|close|read|write|lseek|stat|"
                r"fstat|mkdir|rmdir|unlink|getcwd|chdir)\b"),   "<unistd.h>"),
    (re.compile(r"\b(socket|connect|bind|listen|accept|send|recv|"
                r"setsockopt|getsockopt|htons|ntohs|inet_addr)\b"), "<sys/socket.h>"),
    (re.compile(r"\b(pthread_create|pthread_join|pthread_mutex_lock|"
                r"pthread_mutex_unlock|pthread_t|pthread_mutex_t)\b"), "<pthread.h>"),
]

# Common forward-declaration stubs for types that may appear in function signatures
_TYPE_STUBS = """\
/* --- Forward declarations (auto-generated) --- */
#ifndef _STUBS_DEFINED
#define _STUBS_DEFINED
typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;
typedef unsigned long  u64;
typedef signed   char  s8;
typedef signed   short s16;
typedef signed   int   s32;
typedef signed   long  s64;
#endif
"""

# File header banner
_BANNER = """\
/* ==========================================================
 * Auto-generated source file for vulnerability analysis.
 * DO NOT EDIT — regenerate via source_file_generator.py
 * ========================================================== */
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    func: dict,
    out_dir: Union[str, Path],
    dataset: str = "unknown",
    add_main_stub: bool = False,
) -> Path:
    """
    Write a self-contained .c source file for the given function dict.

    Args:
        func:           Function dict from function_extractor.py.
        out_dir:        Root output directory (files go in out_dir/dataset/).
        dataset:        Sub-folder name, e.g. "bigvul" or "megavul".
        add_main_stub:  If True, append a stub main() so the file compiles
                        as a standalone binary (useful for clang/gcc checks).

    Returns:
        Path to the written .c file.
    """
    out_path = _make_output_path(func["id"], out_dir, dataset)

    # Skip re-generation if file already exists and content matches
    content = _build_source(func, add_main_stub)
    if out_path.exists():
        existing_hash = _sha256(out_path.read_text(encoding="utf-8"))
        new_hash      = _sha256(content)
        if existing_hash == new_hash:
            log.debug(f"[{func['id']}] Source unchanged — skipping write")
            return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    log.info(f"[{func['id']}] Written → {out_path}")
    return out_path


def generate_batch(
    funcs: list[dict],
    out_dir: Union[str, Path],
    dataset: str = "unknown",
    add_main_stub: bool = False,
) -> list[Path]:
    """
    Generate source files for a list of function dicts.

    Returns:
        List of Paths in the same order as input.
    """
    paths = []
    for func in funcs:
        try:
            p = generate(func, out_dir, dataset, add_main_stub)
            paths.append(p)
        except Exception as e:
            log.error(f"Failed to generate source for {func.get('id', '?')}: {e}")
            paths.append(None)
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_source(func: dict, add_main_stub: bool) -> str:
    """Assemble the complete .c file content."""
    code = func["code"]

    headers = _infer_headers(code)
    header_block = "\n".join(f"#include {h}" for h in sorted(headers))

    parts = [
        _BANNER,
        header_block,
        "",
        _TYPE_STUBS,
        "",
        f"/* --- Function: {func['name']} --- */",
        f"/* Source: {func.get('source_file', 'dataset row')} */",
        f"/* Original lines: {func['start_line']}–{func['end_line']} */",
        "",
        code,
        "",
    ]

    if add_main_stub:
        parts += [
            "/* --- Stub main (for standalone compilation only) --- */",
            "int main(void) {",
            f"    /* {func['name']}() would be called here */",
            "    return 0;",
            "}",
            "",
        ]

    return "\n".join(parts)


def _infer_headers(code: str) -> set[str]:
    """
    Scan code for known symbols and return required headers.
    Always includes stdio.h and stdlib.h as a baseline.
    """
    headers = {"<stdio.h>", "<stdlib.h>"}
    for pattern, header in _HEADER_RULES:
        if pattern.search(code):
            headers.add(header)
    return headers


def _make_output_path(func_id: str, out_dir: Union[str, Path], dataset: str) -> Path:
    """Build the output file path: out_dir/dataset/<func_id>.c"""
    safe_id = re.sub(r"[^\w\-]", "_", func_id)   # sanitise for filesystem
    return Path(out_dir) / dataset / f"{safe_id}.c"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python source_file_generator.py <func_json_file> <out_dir>")
        sys.exit(1)

    func_data = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])

    if isinstance(func_data, list):
        paths = generate_batch(func_data, out)
        for p in paths:
            print(p)
    else:
        print(generate(func_data, out))