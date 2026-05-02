# plot_results.py
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from sklearn.metrics import roc_curve, auc

print("Generating result plots...")

# ────────────────────────────────────────────────
# Load real F1 values from repair_metrics.json
# ────────────────────────────────────────────────
METRICS_FILE = "/mnt/data/junaid/linevul/linevul/outputs/repair_metrics.json"

orig_f1 = 0.0
repaired_f1 = 0.0

if os.path.exists(METRICS_FILE):
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        orig_f1 = metrics["original"]["F1"]
        repaired_f1 = metrics["repaired_default"]["F1"]
        print(f"Using real F1: Original = {orig_f1:.4f}, Repaired = {repaired_f1:.4f}")
    except Exception as e:
        print(f"Error loading metrics: {e}")
else:
    print(f"Metrics file not found → skipping F1 plot")

delta_f1 = repaired_f1 - orig_f1

# ────────────────────────────────────────────────
# F1 Before vs After Bar
# ────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
metrics = ['Original F1', 'Repaired F1']
values = [orig_f1, repaired_f1]
colors = ['lightcoral', 'lightgreen']

bars = plt.bar(metrics, values, color=colors)
plt.ylabel('F1 Score')
plt.title('F1 Score Before vs After Repair')
plt.ylim(0, max(repaired_f1 + 0.1, 0.4))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.4f}', ha='center', va='bottom')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('f1_before_after_bar.png')
plt.close()
print("Saved: f1_before_after_bar.png")

# ────────────────────────────────────────────────
# ROC for Trust Detection (real data with inversion)
# ────────────────────────────────────────────────
TRUST_FILE = "/mnt/data/junaid/linevul/linevul/outputs/trust_scores.json"

if os.path.exists(TRUST_FILE):
    try:
        with open(TRUST_FILE, "r") as f:
            trust_data = json.load(f)
        y_true = trust_data["y_true"]
        y_scores = trust_data["y_scores"]

        # Invert to risk scores for positive class (untrustworthy = 1)
        # Higher risk = more likely positive
        y_risk_scores = [1 - s for s in y_scores]

        # Check for both classes
        unique_classes = set(y_true)
        if len(unique_classes) < 2:
            print("Warning: Only one class in y_true → ROC is not meaningful yet")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_risk_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC for Trust Detection')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig('trust_roc.png')
            plt.close()
            print("Saved: trust_roc.png (real AUC plotted)")
    except Exception as e:
        print(f"Error plotting ROC: {e}")
else:
    print("No trust_scores.json found → skipping ROC plot")
    print("Run aggregate_score.py on multiple samples to generate it.")

print("\nAll plots generated. Check files in current directory.")