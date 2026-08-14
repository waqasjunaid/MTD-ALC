# -*- coding: utf-8 -*-
"""
regen_armA.py  (v2 - fixes stale source_file path)

Regenerates the repair patch (Arm A = your real diagnosis-guided method)
for the truly-vulnerable flagged samples, reusing your existing pipeline
scripts unchanged, and FIXING the stale source_file path so the LLM
actually sees the real function (Bug 1 from the test run).

Per sample:
  1. write sample_pred.json for MTD
  2. run mtd
  3. write the REAL source code to a local .c file and rewrite
     mtd_result.json['source_file'] to point at it  <-- Bug 1 fix
  4. run alc -> diagnosis -> repair
  5. verify repair actually produced a patch (patch_generated == True)
     and archive repaired_code.c -> patches_armA/{sample_id}.c
     Records failures so you can see real vs placeholder patches.

Resumable: skips sample_ids already archived.

Usage:
  python regen_armA.py \
    --records diagnosis_records.jsonl \
    --preprocessed /mnt/.../data/bigvul_preprocessed.jsonl \
    --outdir outputs/bigvul \
    --archive patches_armA \
    --srcdir armA_sources \
    --root . \
    [--limit 5]
"""
import argparse, json, subprocess, sys, shutil
from pathlib import Path


def load_preprocessed(path):
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_id[str(r.get("id", i))] = r
    return by_id


def get_source_code(pre_rec, fallback_rec):
    """Pull the real function text. Prefer preprocessed func.code, then the
    diagnosis_records vulnerable_func."""
    func = pre_rec.get("func", {})
    if isinstance(func, dict):
        for k in ("code", "func_before", "source", "text", "body"):
            if func.get(k):
                return str(func[k])
    if isinstance(func, str) and func.strip():
        return func
    # fallback to the record from diagnosis_records.jsonl
    if fallback_rec and fallback_rec.get("vulnerable_func"):
        return str(fallback_rec["vulnerable_func"])
    return ""


def run(cmd):
    return subprocess.run(cmd).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--preprocessed", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--srcdir", default="armA_sources")
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive); archive.mkdir(parents=True, exist_ok=True)
    srcdir = Path(args.srcdir).resolve(); srcdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    MTD  = str(root / "mtd" / "run_mtd.py")
    ALC  = str(root / "alc" / "run_alc.py")
    DIAG = str(root / "diagnosis" / "run_diagnosis.py")
    REP  = str(root / "repair" / "run_repair.py")

    pre = load_preprocessed(args.preprocessed)
    records = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[:args.limit]

    done = placeholder = failed = skipped = 0
    fail_log = open("armA_failures.txt", "a", encoding="utf-8")

    for rec in records:
        sid = str(rec["id"])
        dest = archive / f"{sid}.c"
        if dest.exists():
            skipped += 1
            continue

        pr = pre.get(sid)
        if pr is None:
            print(f"[{sid}] no preprocessed record"); failed += 1
            fail_log.write(f"{sid}\tno_preprocessed\n"); continue

        # --- write sample_pred.json for MTD ---
        (outdir / "sample_pred.json").write_text(json.dumps({
            "sample_id": sid, "dataset": "bigvul",
            "label": pr.get("label", 1),
            "file": pr.get("source_file", ""),
            "suspicious_lines": pr.get("suspicious_line_numbers", []),
            "func": pr.get("func", {}),
            "line_map": pr.get("line_map", {}),
        }), encoding="utf-8")

        # --- Step 1: MTD ---
        if not run([py, MTD, "--out", str(outdir)]):
            print(f"[{sid}] MTD failed"); failed += 1
            fail_log.write(f"{sid}\tmtd_failed\n"); continue

        # --- Bug 1 fix: write REAL source and repoint mtd_result.source_file ---
        code = get_source_code(pr, rec)
        if not code:
            print(f"[{sid}] no source code available"); failed += 1
            fail_log.write(f"{sid}\tno_source\n"); continue
        src_path = srcdir / f"{sid}.c"
        src_path.write_text(code, encoding="utf-8")

        mtd_json = outdir / "mtd_result.json"
        try:
            m = json.loads(mtd_json.read_text(encoding="utf-8"))
            m["source_file"] = str(src_path)
            mtd_json.write_text(json.dumps(m, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[{sid}] could not patch mtd_result: {e}"); failed += 1
            fail_log.write(f"{sid}\tmtd_patch_failed\n"); continue

        # --- Step 2-4: ALC -> Diagnosis -> Repair ---
        ok = (run([py, ALC, "--out", str(outdir)]) and
              run([py, DIAG, "--out", str(outdir), "--force"]) and
              run([py, REP, "--out", str(outdir), "--force"]))

        # --- verify a REAL patch was produced ---
        rr = outdir / "repair_result.json"
        patch_ok = False
        if rr.exists():
            try:
                d = json.loads(rr.read_text(encoding="utf-8"))
                patch_ok = bool(
                    d.get("stage1_vulnerability_fix", {}).get("patch_generated")
                )
            except Exception:
                patch_ok = False

        patch = outdir / "repaired_code.c"
        if ok and patch.exists() and patch_ok:
            shutil.copyfile(patch, dest)
            done += 1
            print(f"[{sid}] OK real patch -> {dest}  ({done} done)")
        elif ok and patch.exists():
            # produced output but Stage-1 fix failed (placeholder) -> DON'T archive
            placeholder += 1
            fail_log.write(f"{sid}\tplaceholder_patch_ok_false\n")
            print(f"[{sid}] PLACEHOLDER (Stage-1 fix failed) - not archived")
        else:
            failed += 1
            fail_log.write(f"{sid}\trepair_failed\n")
            print(f"[{sid}] FAILED")

    fail_log.close()
    print(f"\nDONE. real={done}  placeholder={placeholder}  "
          f"failed={failed}  skipped={skipped}  "
          f"archived_total={len(list(archive.glob('*.c')))}")
    print("Failures logged to armA_failures.txt")


if __name__ == "__main__":
    main()