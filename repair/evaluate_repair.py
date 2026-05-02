# repair/evaluate_repair.py
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel
import torch

# ------------------- CONFIG -------------------
ORIG_MODEL_PATH = "/mnt/data/junaid/linevul/linevul/outputs/line_clf.joblib"
REPAIRED_MODEL_PATH = "/mnt/data/junaid/linevul/linevul/outputs/line_clf_repaired.joblib"
CODEBERT_PATH = "/mnt/data/junaid/linevul/codebert/"

# Load original classifier (CodeBERT embeddings)
orig_bundle = joblib.load(ORIG_MODEL_PATH)
if isinstance(orig_bundle, dict):
    if "clf" in orig_bundle:
        orig_clf = orig_bundle["clf"]
    else:
        raise KeyError(f"Original model dict keys: {list(orig_bundle.keys())}")
else:
    orig_clf = orig_bundle

print("[EVAL] Original classifier loaded successfully")

# Load repaired model (TF-IDF + LR)
repaired_bundle = joblib.load(REPAIRED_MODEL_PATH)
if isinstance(repaired_bundle, dict) and "pipeline" in repaired_bundle:
    repaired_model = repaired_bundle["pipeline"]
else:
    raise ValueError("Repaired model must contain 'pipeline' key")

# Load CodeBERT for original predictions
tokenizer = AutoTokenizer.from_pretrained(CODEBERT_PATH)
codebert_model = AutoModel.from_pretrained(CODEBERT_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
codebert_model.to(device)
codebert_model.eval()

def get_codebert_embedding(text: str):
    if not text.strip():
        return np.zeros((1, 768))  # Empty → zero embedding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = codebert_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

# Load validation set
val_df = pd.read_csv("/mnt/data/junaid/linevul/data/big-vul_dataset/val.csv")
X_val_raw = val_df["processed_func"].fillna("").tolist()
y_val = val_df["target"].tolist()

# ------------------- PREDICT ORIGINAL -------------------
print("Predicting with original model (CodeBERT embeddings)...")
y_pred_orig = []
for text in X_val_raw:
    emb = get_codebert_embedding(text)
    pred = orig_clf.predict(emb)[0]
    y_pred_orig.append(pred)

# ------------------- PREDICT REPAIRED -------------------
print("Predicting with repaired model (TF-IDF)...")
y_pred_repaired = repaired_model.predict(X_val_raw)

# Get raw probabilities for threshold tuning
proba_repaired = repaired_model.predict_proba(X_val_raw)[:, 1]  # prob of vulnerable (class 1)

# Lower threshold set to 0.3
threshold = 0.3
y_pred_repaired_lower = (proba_repaired > threshold).astype(int)

# ------------------- METRICS -------------------
print("\nOriginal Model (CodeBERT + LR):")
print(f"F1:       {f1_score(y_val, y_pred_orig):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_orig):.4f}")
print(f"Recall:   {recall_score(y_val, y_pred_orig):.4f}\n")

print("Repaired Model (TF-IDF + LR, default threshold 0.5):")
print(f"F1:       {f1_score(y_val, y_pred_repaired):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_repaired):.4f}")
print(f"Recall:   {recall_score(y_val, y_pred_repaired):.4f}\n")

print(f"Repaired Model (lower threshold {threshold}):")
print(f"F1:       {f1_score(y_val, y_pred_repaired_lower):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_repaired_lower):.4f}")
print(f"Recall:   {recall_score(y_val, y_pred_repaired_lower):.4f}\n")

# ------------------- REPAIR RATE -------------------
num_repaired = 0
total_vulnerable_pred = 0

for orig_pred, repaired_pred in zip(y_pred_orig, y_pred_repaired):
    if orig_pred == 1:  # originally predicted vulnerable
        total_vulnerable_pred += 1
        if repaired_pred == 0:  # now predicted benign
            num_repaired += 1

repair_rate = num_repaired / total_vulnerable_pred if total_vulnerable_pred > 0 else 0
print(f"Repair rate (originally vulnerable → benign after repair):")
print(f"  {num_repaired}/{total_vulnerable_pred} samples ({repair_rate*100:.1f}%)")

# ------------------- SAVE METRICS -------------------
metrics_output = {
    "original": {
        "F1": float(f1_score(y_val, y_pred_orig)),
        "Precision": float(precision_score(y_val, y_pred_orig)),
        "Recall": float(recall_score(y_val, y_pred_orig))
    },
    "repaired_default": {
        "F1": float(f1_score(y_val, y_pred_repaired)),
        "Precision": float(precision_score(y_val, y_pred_repaired)),
        "Recall": float(recall_score(y_val, y_pred_repaired))
    },
    "repaired_lower_threshold": {
        "threshold": threshold,
        "F1": float(f1_score(y_val, y_pred_repaired_lower)),
        "Precision": float(precision_score(y_val, y_pred_repaired_lower)),
        "Recall": float(recall_score(y_val, y_pred_repaired_lower))
    },
    "repair_rate": {
        "num_repaired": num_repaired,
        "total_vulnerable_pred": total_vulnerable_pred,
        "rate": float(repair_rate)
    }
}

OUTPUT_METRICS_FILE = "/mnt/data/junaid/linevul/linevul/outputs/repair_metrics.json"
with open(OUTPUT_METRICS_FILE, "w") as f:
    json.dump(metrics_output, f, indent=4)

print(f"\nMetrics saved to: {OUTPUT_METRICS_FILE}")