# repair/inference_correction.py

def corrected_score(raw_learned: float, diagnosis: list) -> float:
    """
    Adjust the raw benign probability based on diagnosis entries.
    Higher penalty for high-impact (large delta) or dangerous types.
    """
    if not diagnosis:
        return raw_learned  # no diagnosis → no change

    total_penalty = 0.0
    max_delta = max(d.get("delta", 0.0) for d in diagnosis) if diagnosis else 0.0

    for entry in diagnosis:
        delta = entry.get("delta", 0.0)
        diag_type = entry.get("type", "unknown")

        # Type-based penalty multiplier (stronger for dangerous types)
        type_penalty = 1.0
        if diag_type in ["dead_danger", "api_mask"]:
            type_penalty = 1.8   # higher penalty for dangerous
        elif diag_type in ["comment_out", "token_remove"]:
            type_penalty = 1.3
        elif diag_type in ["var_rename"]:
            type_penalty = 0.9   # lower penalty for rename

        # Scale penalty by delta size
        weighted_delta = delta * type_penalty
        total_penalty += weighted_delta

    # Average penalty + cap it
    avg_penalty = total_penalty / len(diagnosis) if diagnosis else 0.0
    penalty = min(avg_penalty * 2.5, 0.50)  # cap max penalty at 50%

    # Apply correction (lower the benign prob if penalty is high)
    corrected = raw_learned * (1.0 - penalty)

    # Never go below 0.05 or above raw
    corrected = max(0.05, min(raw_learned, corrected))

    return corrected