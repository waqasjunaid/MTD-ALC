#!/usr/bin/env python3
"""
task_ablation_mcnemar.py

Paired significance test (McNemar's exact) between Task-1-only and the
full multi-task model's predictions on the SAME 3,772 test samples,
using the per-sample predictions saved by task_ablation.py.

Usage:
    python task_ablation_mcnemar.py task_ablation_per_sample.json
"""
import json
import math
import sys


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    y = d["y_test"]
    p_t1 = d["task1_only"]
    p_full = d["full"]

    assert len(y) == len(p_t1) == len(p_full)

    both_correct = only_t1_correct = only_full_correct = both_wrong = 0
    for yi, a, b in zip(y, p_t1, p_full):
        a_correct = (a == yi)
        b_correct = (b == yi)
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            only_t1_correct += 1
        elif b_correct and not a_correct:
            only_full_correct += 1
        else:
            both_wrong += 1

    print(f"n = {len(y)}")
    print(f"\n2x2 contingency table (correct prediction?):")
    print(f"                    full: correct   full: wrong")
    print(f"  T1-only: correct  {both_correct:>13}   {only_t1_correct:>13}")
    print(f"  T1-only: wrong    {only_full_correct:>13}   {both_wrong:>13}")

    b, c = only_t1_correct, only_full_correct
    p_value = exact_mcnemar_pvalue(b, c)
    print(f"\nDiscordant pairs: T1-only-only-correct={b}, full-only-correct={c}")
    print(f"McNemar's exact test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("-> Statistically significant at alpha=0.05.")
    else:
        print("-> NOT statistically significant at alpha=0.05.")


if __name__ == "__main__":
    main()
