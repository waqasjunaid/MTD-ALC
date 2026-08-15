#!/usr/bin/env python3
"""
patch_vulrepair_data.py

Adds --train_data_file / --eval_data_file / --test_data_file CLI args
to vulrepair_main.py, and wires them into TextDataset (replacing the
hardcoded datasets.load_dataset("MickyMike/cvefixes_bigvul", ...) calls),
so we can train/evaluate on our own 548-sample data instead of the
original (differently-sized, differently-split) benchmark.

Uses exact line-number verification (from `grep -n` against the real
file) before any edit, aborting with a clear diagnostic on any mismatch
rather than risking a silent bad edit.

Usage:
    python patch_vulrepair_data.py M1_VulRepair_PL-NL/vulrepair_main.py
"""
import sys


def check(lines, line_no, expected_substr):
    actual = lines[line_no - 1]
    if expected_substr not in actual:
        print(f"ABORTED: line {line_no} does not contain expected text.")
        print(f"  expected substring: {expected_substr!r}")
        print(f"  actual line {line_no}: {actual!r}")
        sys.exit(1)


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Verify every anchor line matches exactly what was confirmed via grep -n
    check(lines, 49, 'def __init__(self, tokenizer, args, train_data=None, val_data=None, file_type="train"):')
    check(lines, 56, 'elif file_type == "test":')
    check(lines, 57, 'data = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="test")')
    check(lines, 58, 'sources = data["source"]')
    check(lines, 59, 'labels = data["target"]')
    check(lines, 287, 'parser.add_argument("--output_dir"')
    check(lines, 362, 'if args.do_train:')
    check(lines, 363, 'train_data_whole = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="train")')
    check(lines, 364, 'df = pd.DataFrame({"source"')
    check(lines, 365, 'train_data, val_data = train_test_split(df, test_size=0.1238, random_state=42)')
    check(lines, 371, 'if args.do_test:')
    check(lines, 377, "test_dataset = TextDataset(tokenizer, args, file_type='test')")

    print("All 12 anchor lines verified. Applying edits...")

    with open(path + ".bak", "w", encoding="utf-8") as f:
        f.writelines(lines)

    # --- Edit 1: __init__ signature (line 49, index 48) -- add test_data param
    lines[48] = ('    def __init__(self, tokenizer, args, train_data=None, val_data=None, '
                 'test_data=None, file_type="train"):\n')

    # --- Edit 2: lines 56-59 (indices 55-58) -- use test_data instead of
    # the hardcoded HuggingFace dataset call
    lines[55:59] = [
        '        elif file_type == "test":\n',
        '            sources = test_data["source"].tolist()\n',
        '            labels = test_data["target"].tolist()\n',
    ]
    # NOTE: this replaces 4 lines with 3, so everything from here on shifts
    # by -1 relative to the original grep -n line numbers. All remaining
    # edits below account for this shift.

    # --- Edit 3: argparse -- insert new args before the original line 287
    argparse_insert_idx = None
    for i, line in enumerate(lines):
        if 'parser.add_argument("--output_dir"' in line:
            argparse_insert_idx = i
            break
    if argparse_insert_idx is None:
        print("ABORTED: could not re-locate --output_dir argparse line after edit 2.")
        sys.exit(1)
    new_args = (
        '    parser.add_argument("--train_data_file", default=None, type=str,\n'
        '                        help="Our own train CSV (source,target columns).")\n'
        '    parser.add_argument("--eval_data_file", default=None, type=str,\n'
        '                        help="Our own val CSV (source,target columns).")\n'
        '    parser.add_argument("--test_data_file", default=None, type=str,\n'
        '                        help="Our own test CSV (source,target columns).")\n'
    )
    lines.insert(argparse_insert_idx, new_args)

    # --- Edit 4: do_train block -- replace HuggingFace loading with our CSVs
    do_train_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "if args.do_train:":
            do_train_idx = i
            break
    if do_train_idx is None:
        print("ABORTED: could not re-locate 'if args.do_train:' after prior edits.")
        sys.exit(1)
    old_block = lines[do_train_idx + 1: do_train_idx + 4]
    expected_markers = ["datasets.load_dataset", 'df = pd.DataFrame({"source"', "train_test_split"]
    for line, marker in zip(old_block, expected_markers):
        if marker not in line:
            print(f"ABORTED: expected marker {marker!r} in do_train block, "
                  f"found: {line!r}")
            sys.exit(1)
    new_train_block = [
        '        train_data = pd.read_csv(args.train_data_file)\n',
        '        val_data = pd.read_csv(args.eval_data_file)\n',
    ]
    lines[do_train_idx + 1: do_train_idx + 4] = new_train_block

    # --- Edit 5: do_test block -- load our test CSV and pass as test_data
    do_test_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "if args.do_test:":
            do_test_idx = i
            break
    if do_test_idx is None:
        print("ABORTED: could not re-locate 'if args.do_test:' after prior edits.")
        sys.exit(1)
    for i in range(do_test_idx, min(do_test_idx + 10, len(lines))):
        if "test_dataset = TextDataset(tokenizer, args, file_type='test')" in lines[i]:
            lines[i] = (
                '        test_data = pd.read_csv(args.test_data_file)\n'
                "        test_dataset = TextDataset(tokenizer, args, test_data=test_data, file_type='test')\n"
            )
            break
    else:
        print("ABORTED: could not find the test_dataset= line under do_test.")
        sys.exit(1)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Patched {path} successfully (backup at {path}.bak)")


if __name__ == "__main__":
    main()
