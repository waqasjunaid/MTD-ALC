#!/usr/bin/env python3
"""
patch_linevul_write_preds.py

Moves the write_raw_preds_csv(...) call to BEFORE generate_result_df(...),
since generate_result_df crashes (KeyError: 'flaw_line_index') on a
minimal 2-column CSV, preventing write_raw_preds_csv from ever running.
write_raw_preds_csv only needs `args` and `y_preds`, both already
available earlier, so this reordering is safe.

Usage:
    python patch_linevul_write_preds.py linevul_main.py
"""
import sys

TARGET_CALL_LINE = "    result_df = generate_result_df(logits, y_trues, y_preds, args)\n"
IF_LINE = "    if args.write_raw_preds:\n"
CALL_LINE = "        write_raw_preds_csv(args, y_preds)\n"


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    try:
        target_idx = lines.index(TARGET_CALL_LINE)
    except ValueError:
        print(f"ABORTED: could not find the exact line:\n  {TARGET_CALL_LINE!r}")
        print("Paste `grep -n \"result_df = generate_result_df\" linevul_main.py` "
              "so I can adjust for the exact formatting.")
        sys.exit(1)

    try:
        if_idx = lines.index(IF_LINE)
        call_idx = lines.index(CALL_LINE)
    except ValueError:
        print("ABORTED: could not find the write_raw_preds block exactly.")
        print(f"  looking for: {IF_LINE!r} and {CALL_LINE!r}")
        sys.exit(1)

    if call_idx != if_idx + 1:
        print(f"ABORTED: expected the write_raw_preds_csv call immediately "
              f"after the if-statement, but found them at indices {if_idx} "
              f"and {call_idx}. Manual check needed.")
        sys.exit(1)

    if if_idx < target_idx:
        print("The write_raw_preds block is already before generate_result_df "
              "-- no change needed.")
        return

    # backup
    with open(path + ".bak", "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Remove the 2-line if-block from its current position
    block = lines[if_idx:if_idx + 2]
    del lines[if_idx:if_idx + 2]

    # Re-find target_idx (may have shifted if it was after the removed block --
    # it wasn't, since if_idx > target_idx was just confirmed, so target_idx
    # is unchanged)
    lines[target_idx:target_idx] = block

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Patched {path} successfully (backup at {path}.bak)")
    print("write_raw_preds_csv(...) now runs BEFORE generate_result_df(...)")


if __name__ == "__main__":
    main()
