import pandas as pd
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import numpy as np
from .inject_deadcode import inject_dead_code

# Load original training data
df = pd.read_csv("/mnt/data/junaid/linevul/data/big-vul_dataset/train.csv")
X_orig = df["processed_func"].fillna("").tolist()
y_orig = df["target"].tolist()

# Load counterfactuals + dead code injections
repairs = json.load(open("/mnt/data/junaid/linevul/linevul/outputs/counterfactual_repairs.json"))

X_aug = []
y_aug = []

for r in repairs:
    repaired = r["repaired"]
    X_aug.append(repaired)
    y_aug.append(0)  # Treat repaired as benign

    # Also add dead-code version
    dead = inject_dead_code(repaired)  # from inject_deadcode
    X_aug.append(dead)
    y_aug.append(0)

# Combine original + augmented
X = X_orig + X_aug
y = y_orig + y_aug

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Train pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=50000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight_dict))
])

print("Training repaired model...")
pipeline.fit(X, y)

# Save
joblib.dump(
    {"type": "tfidf", "pipeline": pipeline},
    "/mnt/data/junaid/linevul/linevul/outputs/line_clf_repaired.joblib"
)

print("Repaired model saved.")

# Optional: quick self-evaluation (better to do on real val set)
# y_pred = pipeline.predict(X_orig)
# print("F1 on original data:", f1_score(y_orig, y_pred))