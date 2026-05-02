# diagnosis/causal_probe.py
from diagnosis.model_probe import predict_full_prob
from diagnosis.perturbations import all_perturbations  # ← THIS WAS MISSING!


def causal_effects(full_code_list: list, target_line_no: int, original_line: str):
    """
    Perturb one line and re-predict on the full function.
    Returns list of effect dicts.
    """
    base_code = "\n".join(full_code_list)
    base_prob = predict_full_prob(base_code)

    print(f"\n[CAUSAL] Line {target_line_no}: {original_line.strip()[:100]}")
    print(f"         Base benign prob: {base_prob:.4f}")

    effects = []
    for p in all_perturbations(original_line):
        try:
            perturbed_lines = full_code_list.copy()
            perturbed_lines[target_line_no] = p["code"]  # 1-indexed → list index = target_line_no
            new_code = "\n".join(perturbed_lines)
            new_prob = predict_full_prob(new_code)
            delta = abs(base_prob - new_prob)

            print(f"         → {p['type']:12} {p['feature']:20} delta = {delta:.4f} (new prob: {new_prob:.4f})")

            effects.append({
                **p,
                "base": base_prob,
                "new": new_prob,
                "delta": delta
            })
        except Exception as e:
            print(f"         [ERROR] Perturbation '{p['type']}' failed: {e}")
            continue

    return effects