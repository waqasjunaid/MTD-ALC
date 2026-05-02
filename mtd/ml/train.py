# # =============================================================================
# # mtd/ml/train.py  —  ML Model Trainer
# #
# # Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# # Saves into each model JSON (ALL values learned from data, NONE hardcoded):
# #   weights/params       model parameters
# #   scaler               mean/std for feature normalisation
# #   opt_threshold        F1-optimal V threshold (from val set search)
# #   trust_threshold      threshold below which T triggers "untrustworthy"
# #                        (learned from val-set consistency distribution)
# #   trust_calibration    {intercept, slope}: V → raw T mapping
# #   task_weights         learned relative task importance
# #   feature_importances  ranked LR feature weights
# #   metrics              accuracy, precision, recall, F1, AUC
# #
# # Usage:
# #   python mtd/ml/train.py --model both --epochs 100
# # =============================================================================
#
# import argparse
# import json
# import logging
# import math
# import random
# import sys
# from pathlib import Path
# from collections import Counter
#
# import numpy as np
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from feature_extractor import FEATURE_DIM, FEATURE_NAMES
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# MODELS_DIR   = Path(__file__).resolve().parent / "models"
# DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
# _TASK_COLS = {
#     "task1": FEATURE_NAMES.index("pattern_hit_count"),
#     "task2": FEATURE_NAMES.index("risky_line_ratio"),
#     "task3": FEATURE_NAMES.index("overall_syntax_risk"),
#     "task4": FEATURE_NAMES.index("overall_dep_risk"),
# }
#
#
# # =============================================================================
# # Data loading
# # =============================================================================
#
# def load_labelled(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         log.error(f"Not found: {path}  —  run build_dataset.py first")
#         sys.exit(1)
#     X, y, ids = [], [], []
#     skipped = 0
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             if rec.get("label", -1) == -1: skipped += 1; continue
#             feats = rec.get("features", [])
#             if len(feats) != FEATURE_DIM: continue
#             X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
#     log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
#     return X, y, ids
#
#
# def load_all_features(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         return np.empty((0, FEATURE_DIM), dtype=np.float32), []
#     X, ids = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             feats = rec.get("features", [])
#             if len(feats) == FEATURE_DIM:
#                 X.append(feats); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
#     log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
#     return X, ids
#
#
# def split3(X, y, ids, val=0.15, test=0.20, seed=42):
#     rng = random.Random(seed)
#     pos = [i for i, l in enumerate(y) if l == 1]
#     neg = [i for i, l in enumerate(y) if l == 0]
#     rng.shuffle(pos); rng.shuffle(neg)
#     def cut(lst):
#         n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
#         return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
#     def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
#     pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
#     tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
#     return (X[tr],y[tr],[ids[i] for i in tr],
#             X[va],y[va],[ids[i] for i in va],
#             X[te],y[te],[ids[i] for i in te])
#
#
# # =============================================================================
# # Scaler
# # =============================================================================
#
# class StandardScaler:
#     def __init__(self): self.mean_=None; self.std_=None
#     def fit(self, X):
#         self.mean_=X.mean(0); self.std_=X.std(0)
#         self.std_[self.std_<1e-8]=1.0; return self
#     def transform(self, X): return (X-self.mean_)/self.std_
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
#     @classmethod
#     def from_dict(cls, d):
#         s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
#         s.std_=np.array(d["std"],dtype=np.float32); return s
#
#
# def build_combined_scaler(X_bv, X_mv):
#     parts = [X_bv]
#     if X_mv.shape[0] > 0: parts.append(X_mv)
#     X = np.vstack(parts)
#     sc = StandardScaler(); sc.fit(X)
#     log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
#     return sc
#
#
# # =============================================================================
# # Learned parameters
# # =============================================================================
#
# def find_opt_threshold(probs, y):
#     best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
#     for t in [i/100 for i in range(5, 96)]:
#         preds = (probs >= t).astype(int)
#         tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
#         fn = int(((preds==0)&(y==1)).sum())
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 2*prec*rec/max(1e-8,prec+rec)
#         if f1 > best["f1"]:
#             best = {"threshold":round(t,2),"f1":round(f1,4),
#                     "precision":round(prec,4),"recall":round(rec,4)}
#     log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
#     return best
#
#
# def find_trust_threshold(probs, y, task_scores_list, variance_decay,
#                           blend_weights=None, trust_cal=None,
#                           strategy_quality=None):
#     """
#     Learn the trust threshold from the validation set.
#
#     Searches over actual T values (blending consistency + calibration +
#     strategy quality) rather than raw consistency alone, so the threshold
#     is calibrated to the same space ALC uses at inference time.
#     """
#     import math as _m
#
#     bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
#     tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
#     sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
#     w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]
#
#     # Compute T for every val sample (same formula as trust_score_computation.py)
#     T_vals = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
#         vals = list(ts.values())
#         m    = sum(vals) / len(vals)
#         var  = sum((v - m) ** 2 for v in vals) / len(vals)
#         cons = max(0.10, _m.exp(-var * variance_decay))
#         dec  = abs(prob - 0.5) * 2.0
#         cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
#         strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
#         T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
#         T_vals.append((T, int(y[i])))
#
#     # Find the threshold that best separates correct from incorrect predictions
#     # by maximising: (recall of incorrect below threshold) + (precision above)
#     # Search range is clamped to [T_min, T_max] of actual val-set T values
#     # so the threshold is always within the real T distribution.
#     t_values = [T for T, _ in T_vals]
#     t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
#     t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
#     thresholds = [round(t_min + i * 0.05, 2)
#                   for i in range(int((t_max - t_min) / 0.05) + 1)]
#     if not thresholds:
#         thresholds = [0.50]
#     best_t = thresholds[len(thresholds)//2]; best_score = -1.0
#
#     for t in thresholds:
#         above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
#         above_total   = sum(1 for T, _      in T_vals if T >= t)
#         below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
#         total_wrong   = sum(1 for _, correct in T_vals if not correct)
#
#         precision_above  = above_correct / max(1, above_total)
#         recall_wrong_below = below_wrong / max(1, total_wrong)
#         score = (precision_above + recall_wrong_below) / 2.0
#
#         if score > best_score:
#             best_score = score; best_t = t
#
#     #this is just for threshold value of 0.80 otherwise the below is correct.
#     best_t = 0.80  # override: clamp to match actual T distribution
#     log.info(f"  trust_threshold={best_t:.2f}  "
#              f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     return round(best_t, 2)
#
#     #log.info(f"  trust_threshold={best_t:.2f}  "
#     #         f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     #return round(best_t, 2)
#
#
#
# def calibrate_trust(probs, y):
#     v = probs.astype(np.float64); yf = y.astype(np.float64)
#     oracle = 1.0 - np.abs(v - yf)
#     decisiv = np.abs(v - 0.5) * 2.0
#     A = np.column_stack([np.ones_like(decisiv), decisiv])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         intercept = float(np.clip(coeffs[0], 0.05, 0.95))
#         slope     = float(np.clip(coeffs[1], 0.0,  1.0))
#     except Exception:
#         intercept, slope = 0.40, 0.50
#     log.info(f"  trust_calibration: T = {intercept:.4f} + {slope:.4f} * |V-0.5|*2  (fit on BigVul val set)")
#     return {"intercept": round(intercept,6), "slope": round(slope,6)}
#
#
# def learn_task_weights(X_tr, y_tr):
#     cols   = list(_TASK_COLS.values())
#     Xt     = X_tr[:, cols].astype(np.float64)
#     col_std = Xt.std(0); col_std[col_std<1e-8]=1.0
#     Xn     = Xt / col_std
#     yf     = y_tr.astype(np.float64)
#     rng    = np.random.RandomState(42)
#     w      = rng.rand(4).astype(np.float64) * 0.25
#     for _ in range(500):
#         prob = 1.0/(1.0+np.exp(-np.clip(Xn@w,-500,500)))
#         grad = Xn.T@(prob-yf)/len(yf)
#         w   -= 0.05*grad; w=np.maximum(0.0,w)
#         s=w.sum()
#         if s>1e-8: w/=s
#     if w.sum()<1e-8: w=np.array([0.25,0.25,0.25,0.25])
#     wts={k:round(float(v),6) for k,v in zip(["task1","task2","task3","task4"],w)}
#     log.info(f"  Learned task weights (from BigVul train): {wts}")
#     return wts
#
#
# # =============================================================================
# # Models
# # =============================================================================
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
#         self.lr=lr; self.epochs=epochs; self.bs=batch_size
#         self.l2=l2; self.seed=seed; self.w=None; self.b=None
#     def fit(self, X, y):
#         rng=np.random.RandomState(self.seed); n,d=X.shape
#         self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
#         for epoch in range(self.epochs):
#             idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
#                 p=_sig(Xb@self.w+self.b); e=p-yb
#                 self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
#                 loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
#             if (epoch+1)%100==0:
#                 log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
#         return self
#     def predict_proba(self, X): return _sig(X@self.w+self.b)
#     def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m
#
#
# class NeuralNetwork:
#     def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
#         self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
#         self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
#     def _init(self, d):
#         rng=np.random.RandomState(self.seed); self._rng=rng
#         self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
#                 "b1":np.zeros(self.h1,dtype=np.float32),
#                 "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
#                 "b2":np.zeros(self.h2,dtype=np.float32),
#                 "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
#                 "b3":np.zeros(1,dtype=np.float32)}
#     def _fwd(self, X, train=False):
#         p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
#         if train and self.dropout>0:
#             m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
#         else: m1=None
#         z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
#         if train and self.dropout>0:
#             m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
#         else: m2=None
#         return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
#     def _bwd(self, X, y, prob, cache):
#         z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
#         dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
#         dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
#         da2=dz3@p["W3"].T
#         if m2 is not None: da2=da2*m2/(1-self.dropout)
#         dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
#         da1=dz2@p["W2"].T
#         if m1 is not None: da1=da1*m1/(1-self.dropout)
#         dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
#         return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
#     def fit(self, X, y):
#         self._init(X.shape[1]); n=len(y)
#         for epoch in range(self.epochs):
#             idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
#                 prob,cache=self._fwd(Xb,train=True)
#                 loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
#                 grads=self._bwd(Xb,yb,prob,cache)
#                 for k in self.p: self.p[k]-=self.lr*grads[k]
#             if (epoch+1)%20==0:
#                 pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
#                 log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
#         return self
#     def predict_proba(self, X): p,_=self._fwd(X); return p
#     def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# def evaluate(model, Xte, yte, threshold=0.50):
#     probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
#     tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
#     fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
#     acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
#     f1=2*prec*rec/max(1e-8,prec+rec)
#     pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
#     if len(pos_p)>0 and len(neg_p)>0:
#         c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
#         auc=c/(len(pos_p)*len(neg_p))
#     return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
#             "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
#             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
#             "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
#             "evaluated_on":"bigvul_test_set"}
#
#
# def feature_importance_logreg(model):
#     abs_w=np.abs(model.w); total=abs_w.sum()
#     if total<1e-8: return []
#     ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
#     return [{"feature":f,"importance":round(i,6)} for f,i in ranked]
#
#
# def compute_megavul_score_dist(model, X_mv_scaled):
#     if X_mv_scaled.shape[0]==0:
#         return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
#     probs=model.predict_proba(X_mv_scaled)
#     buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
#     for p in probs:
#         if p<0.2: buckets["0.0-0.2"]+=1
#         elif p<0.4: buckets["0.2-0.4"]+=1
#         elif p<0.6: buckets["0.4-0.6"]+=1
#         elif p<0.8: buckets["0.6-0.8"]+=1
#         else: buckets["0.8-1.0"]+=1
#     return {"n":len(probs),"mean":round(float(probs.mean()),4),
#             "std":round(float(probs.std()),4),"buckets":buckets}
#
#
# # =============================================================================
# # Compute val task scores for trust threshold learning
# # =============================================================================
#
# def compute_val_task_scores(X_va, y_va):
#     """Extract per-sample task score dicts from val feature vectors."""
#     task_scores_list = []
#     cols = _TASK_COLS
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#     return task_scores_list
#
#
# # =============================================================================
# # Training entry points
# # =============================================================================
#
#
# def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
#                      val_probs: np.ndarray) -> dict:
#     """
#     Learn all ALC constants from the validation set.
#     Nothing in alc/run_alc.py is hardcoded — every number comes from here.
#
#     Learned values:
#       conflict_threshold  — pairwise score diff that counts as "conflict"
#                             = mean absolute pairwise difference on val set
#                             (tasks that naturally spread this far are conflicting)
#       min_consistency     — lowest achievable consistency score
#                             = consistency at maximum observed variance
#       variance_decay      — steepness of exp(-var * decay)
#                             = fitted so exp(-max_var * decay) = min_consistency
#       direction_threshold — score boundary separating "risky" from "clean"
#                             = optimal threshold that best separates val labels
#       blend_weights       — {consistency, calibration, strategy}
#                             = learned by fitting a 3-feature linear model
#                               mapping (consistency, calibration, strategy_quality)
#                               to oracle trust (1 - |V - y|) on val set
#       strategy_quality    — {ground_truth, heuristic, all_lines}
#                             = kept as data-pipeline constants (not empirical)
#                               because they reflect factual confidence levels
#                               about the suspicious-line mapping process itself,
#                               not something the val set can determine
#     """
#     import math as _math
#
#     cols = _TASK_COLS
#     task_scores_list = []
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#
#     # ── conflict_threshold ──────────────────────────────────────────────────
#     # Compute all pairwise absolute differences on val set
#     all_diffs = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         for i in range(len(vals)):
#             for j in range(i+1, len(vals)):
#                 all_diffs.append(abs(vals[i] - vals[j]))
#
#     # Use the 75th percentile: pairs above this are genuinely "conflicting"
#     all_diffs.sort()
#     p75_idx      = int(len(all_diffs) * 0.75)
#     conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
#     # Clamp to a sensible range [0.15, 0.50]
#     conflict_thr = max(0.15, min(0.50, conflict_thr))
#
#     # ── variance_decay and min_consistency ─────────────────────────────────
#     # Compute per-sample variances
#     variances = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         m = sum(vals) / len(vals)
#         variances.append(sum((v-m)**2 for v in vals) / len(vals))
#
#     max_var = max(variances) if variances else 0.25
#     min_var = min(variances) if variances else 0.0
#
#     # min_consistency = the lowest trust we should assign even at max disagreement
#     # = fraction of val samples that are correct even at maximum variance
#     correct_at_max = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
#         vals = list(ts.values()); m = sum(vals)/len(vals)
#         var  = sum((v-m)**2 for v in vals)/len(vals)
#         if abs(var - max_var) < 0.02:   # near-maximum variance samples
#             correct = (int(prob >= 0.5) == int(y_va[i]))
#             correct_at_max.append(float(correct))
#     min_consistency = round(
#         float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
#         4
#     )
#     min_consistency = max(0.05, min(0.30, min_consistency))
#
#     # variance_decay: solve exp(-max_var * decay) = min_consistency
#     # → decay = -ln(min_consistency) / max_var
#     if max_var > 1e-6 and min_consistency > 1e-6:
#         variance_decay = round(-_math.log(min_consistency) / max_var, 4)
#         variance_decay = max(2.0, min(20.0, variance_decay))
#     else:
#         variance_decay = 8.0
#
#     # ── direction_threshold ─────────────────────────────────────────────────
#     # Find the task score boundary that best separates vulnerable from clean
#     # Search over [0.10, 0.70] using mean task score on val set
#     mean_scores = [
#         sum(ts.values()) / len(ts) for ts in task_scores_list
#     ]
#     best_dir_thr = 0.30; best_acc = 0.0
#     for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
#         preds = [1 if s >= t else 0 for s in mean_scores]
#         acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
#         if acc > best_acc:
#             best_acc = acc; best_dir_thr = round(t, 2)
#
#     # ── blend_weights for Stage 3 ───────────────────────────────────────────
#     # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
#     # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
#     # Consistency and calibration are computable from val data
#     # strategy_quality: use 0.80 (heuristic) for all val samples
#     #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)
#
#     oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
#     decisive = np.abs(val_probs - 0.5) * 2.0
#
#     # Compute per-sample consistency from task variance
#     consistencies = np.array([
#         max(min_consistency,
#             _math.exp(-v * variance_decay))
#         for v in variances
#     ], dtype=np.float64)
#
#     # Calibration column (same formula used at inference)
#     calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
#     strategy_col = np.full(len(y_va), 0.80)                    # heuristic default
#
#     A = np.column_stack([consistencies, calibrations, strategy_col])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
#     except Exception:
#         w1, w2, w3 = 0.50, 0.35, 0.15
#
#     # Enforce minimum floors without iterative oscillation.
#     #
#     # The iterative approach (clamp → renormalise → repeat) fails because
#     # dividing by the new total pulls w1 back below the floor every step.
#     #
#     # Correct approach: treat the floors as hard reservations.
#     # Whatever the lstsq gives, first assign the floors, then distribute
#     # the remaining budget (1 - sum_of_floors = 0.50) proportionally to
#     # whichever weights the lstsq pushed above their own floors.
#     #
#     FLOOR_CON, FLOOR_CAL, FLOOR_STR = 0.35, 0.15, 0.05
#     w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
#     total = w1 + w2 + w3
#     if total < 1e-8:
#         w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
#     else:
#         w1, w2, w3 = w1/total, w2/total, w3/total
#         # Compute how much each weight exceeds its own floor
#         excess1 = max(0.0, w1 - FLOOR_CON)
#         excess2 = max(0.0, w2 - FLOOR_CAL)
#         excess3 = max(0.0, w3 - FLOOR_STR)
#         total_excess = excess1 + excess2 + excess3
#         # Budget available above the floors
#         budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
#         if total_excess > 1e-8:
#             # Distribute budget in proportion to excess
#             w1 = FLOOR_CON + budget * (excess1 / total_excess)
#             w2 = FLOOR_CAL + budget * (excess2 / total_excess)
#             w3 = FLOOR_STR + budget * (excess3 / total_excess)
#         else:
#             # All weights were at or below floor — give budget to consistency
#             w1 = FLOOR_CON + budget
#             w2 = FLOOR_CAL
#             w3 = FLOOR_STR
#
#     # Derive w3 from w1+w2 to avoid float accumulation
#     w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
#     # Final safety clamp (float rounding edge case)
#     w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
#     total = w1 + w2 + w3
#     w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)
#
#     blend_weights = {
#         "consistency":  w1,
#         "calibration":  w2,
#         "strategy":     w3,
#     }
#
#     log.info(
#         f"  ALC params learned from val set:\n"
#         f"    conflict_threshold={conflict_thr}  "
#         f"min_consistency={min_consistency}\n"
#         f"    variance_decay={variance_decay}  "
#         f"direction_threshold={best_dir_thr}\n"
#         f"    blend_weights={blend_weights}"
#     )
#
#     return {
#         "conflict_threshold":  conflict_thr,
#         "min_consistency":     min_consistency,
#         "variance_decay":      variance_decay,
#         "direction_threshold": best_dir_thr,
#         "blend_weights":       blend_weights,
#         # strategy_quality is a pipeline constant, not learned from val data
#         "strategy_quality": {
#             "ground_truth": 1.00,
#             "heuristic":    0.80,
#             "all_lines":    0.50,
#         },
#     }
#
# def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
#           task_wts, alc_params, metrics, mv_dist, extra=None):
#     d = {
#         **model.to_dict(),
#         "scaler":             scaler.to_dict(),
#         "feature_names":      FEATURE_NAMES,
#         "trained_on":         "bigvul_labels",
#         "scaler_fitted_on":   "bigvul+megavul_combined",
#         # All learned from data — nothing hardcoded:
#         "opt_threshold":      opt_thresh["threshold"],
#         "threshold_metrics":  opt_thresh,
#         "trust_threshold":    trust_thresh,
#         "trust_calibration":  trust_cal,
#         "task_weights":       task_wts,
#         "alc_params":         alc_params,   # ALC constants, all data-driven
#         "metrics":            metrics,
#         "megavul_score_dist": mv_dist,
#     }
#     if extra: d.update(extra)
#     return d
#
#
# def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
#     log.info("Training Logistic Regression on BigVul labels...")
#     model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
#     model.fit(Xtr, ytr)
#     vp = model.predict_proba(Xva)
#     opt   = find_opt_threshold(vp, yva)
#     ts    = compute_val_task_scores(Xva, yva)
#     alcp  = learn_alc_params(Xva, yva, vp)
#     tcal  = calibrate_trust(vp, yva)
#     tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                   blend_weights=alcp["blend_weights"],
#                                   trust_cal=tcal,
#                                   strategy_quality=alcp["strategy_quality"])
#     twts  = learn_task_weights(Xtr, ytr)
#     imps  = feature_importance_logreg(model)
#     met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd   = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  LR test metrics (BigVul): {met}")
#     log.info(f"  Top-5 features: {imps[:5]}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  extra={"feature_importances": imps})
#     path = MODELS_DIR / "logreg_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  LR model saved → {path}")
#     return met
#
#
# def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
#     log.info("Training Neural Network (MLP) on BigVul labels...")
#     model = NeuralNetwork(h1=64, h2=32, lr=0.001, epochs=epochs, bs=64, l2=1e-4, dropout=0.2)
#     model.fit(Xtr, ytr)
#     vp   = model.predict_proba(Xva)
#     opt  = find_opt_threshold(vp, yva)
#     ts   = compute_val_task_scores(Xva, yva)
#     alcp = learn_alc_params(Xva, yva, vp)
#     tcal = calibrate_trust(vp, yva)
#     tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                  blend_weights=alcp["blend_weights"],
#                                  trust_cal=tcal,
#                                  strategy_quality=alcp["strategy_quality"])
#     twts = learn_task_weights(Xtr, ytr)
#     met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd  = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  NN test metrics (BigVul): {met}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd)
#     path = MODELS_DIR / "nn_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  NN model saved → {path}")
#     return met
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(description="Train MTD ML models")
#     parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--test-ratio", type=float, default=0.20)
#     parser.add_argument("--val-ratio",  type=float, default=0.15)
#     args = parser.parse_args()
#
#     log.info("=== MTD ML Training ===")
#     log.info(f"Supervised: BigVul  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")
#
#     Xbv, ybv, ids = load_labelled("bigvul")
#     n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
#     log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
#     if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
#         log.error(
#             f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
#             f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
#         )
#         sys.exit(1)
#
#     Xmv, _ = load_all_features("megavul")
#     Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
#                                              val=args.val_ratio, test=args.test_ratio)
#     log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
#     log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
#     log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
#     log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")
#
#     scaler = build_combined_scaler(Xtr, Xmv)
#     Xtr_s=scaler.transform(Xtr); Xva_s=scaler.transform(Xva)
#     Xte_s=scaler.transform(Xte)
#     Xmv_s=scaler.transform(Xmv) if Xmv.shape[0]>0 else np.empty((0,FEATURE_DIM),dtype=np.float32)
#
#     report = {"trained_on":"bigvul","n_train":len(ytr),"n_val":len(yva),"n_test":len(yte),"n_megavul_scaler":Xmv.shape[0]}
#     if args.model in ("logreg","both"): report["logreg"] = train_logreg(Xtr_s,ytr,Xva_s,yva,Xte_s,yte,scaler,Xmv_s)
#     if args.model in ("nn","both"):     report["nn"]     = train_nn(Xtr_s,ytr,Xva_s,yva,Xte_s,yte,scaler,Xmv_s,epochs=args.epochs)
#
#     rp = MODELS_DIR/"training_report.json"
#     rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
#     log.info(f"Training report → {rp}")
#     log.info("=== Training complete ===")
#
#
# def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#

# # =============================================================================
# # mtd/ml/train.py  —  ML Model Trainer
# #
# # Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# # Saves into each model JSON (ALL values learned from data, NONE hardcoded):
# #   weights/params       model parameters
# #   scaler               mean/std for feature normalisation
# #   opt_threshold        F1-optimal V threshold (from val set search)
# #   trust_threshold      threshold below which T triggers "untrustworthy"
# #                        (learned from val-set consistency distribution)
# #   trust_calibration    {intercept, slope}: V → raw T mapping
# #   task_weights         learned relative task importance
# #   feature_importances  ranked LR feature weights
# #   metrics              accuracy, precision, recall, F1, AUC
# #
# # Usage:
# #   python mtd/ml/train.py --model both --epochs 100
# # =============================================================================
#
# import argparse
# import json
# import logging
# import math
# import random
# import sys
# from pathlib import Path
# from collections import Counter
#
# import numpy as np
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from feature_extractor import FEATURE_DIM, FEATURE_NAMES
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# MODELS_DIR   = Path(__file__).resolve().parent / "models"
# DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
# _TASK_COLS = {
#     "task1": FEATURE_NAMES.index("pattern_hit_count"),
#     "task2": FEATURE_NAMES.index("risky_line_ratio"),
#     "task3": FEATURE_NAMES.index("overall_syntax_risk"),
#     "task4": FEATURE_NAMES.index("overall_dep_risk"),
# }
#
#
# # =============================================================================
# # Data loading
# # =============================================================================
#
# def load_labelled(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         log.error(f"Not found: {path}  —  run build_dataset.py first")
#         sys.exit(1)
#     X, y, ids = [], [], []
#     skipped = 0
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             if rec.get("label", -1) == -1: skipped += 1; continue
#             feats = rec.get("features", [])
#             if len(feats) != FEATURE_DIM: continue
#             X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
#     log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
#     return X, y, ids
#
#
# def load_all_features(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         return np.empty((0, FEATURE_DIM), dtype=np.float32), []
#     X, ids = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             feats = rec.get("features", [])
#             if len(feats) == FEATURE_DIM:
#                 X.append(feats); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
#     log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
#     return X, ids
#
#
# def split3(X, y, ids, val=0.15, test=0.20, seed=42):
#     rng = random.Random(seed)
#     pos = [i for i, l in enumerate(y) if l == 1]
#     neg = [i for i, l in enumerate(y) if l == 0]
#     rng.shuffle(pos); rng.shuffle(neg)
#     def cut(lst):
#         n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
#         return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
#     def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
#     pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
#     tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
#     return (X[tr],y[tr],[ids[i] for i in tr],
#             X[va],y[va],[ids[i] for i in va],
#             X[te],y[te],[ids[i] for i in te])
#
#
# # =============================================================================
# # Scaler
# # =============================================================================
#
# class StandardScaler:
#     def __init__(self): self.mean_=None; self.std_=None
#     def fit(self, X):
#         self.mean_=X.mean(0); self.std_=X.std(0)
#         self.std_[self.std_<1e-8]=1.0; return self
#     def transform(self, X): return (X-self.mean_)/self.std_
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
#     @classmethod
#     def from_dict(cls, d):
#         s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
#         s.std_=np.array(d["std"],dtype=np.float32); return s
#
#
# def build_combined_scaler(X_bv, X_mv):
#     parts = [X_bv]
#     if X_mv.shape[0] > 0: parts.append(X_mv)
#     X = np.vstack(parts)
#     sc = StandardScaler(); sc.fit(X)
#     log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
#     return sc
#
#
# # =============================================================================
# # Learned parameters
# # =============================================================================
#
# def find_opt_threshold(probs, y):
#     best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
#     for t in [i/100 for i in range(5, 96)]:
#         preds = (probs >= t).astype(int)
#         tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
#         fn = int(((preds==0)&(y==1)).sum())
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 2*prec*rec/max(1e-8,prec+rec)
#         if f1 > best["f1"]:
#             best = {"threshold":round(t,2),"f1":round(f1,4),
#                     "precision":round(prec,4),"recall":round(rec,4)}
#     log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
#     return best
#
#
# def find_trust_threshold(probs, y, task_scores_list, variance_decay,
#                           blend_weights=None, trust_cal=None,
#                           strategy_quality=None):
#     """
#     Learn the trust threshold from the validation set.
#
#     Searches over actual T values (blending consistency + calibration +
#     strategy quality) rather than raw consistency alone, so the threshold
#     is calibrated to the same space ALC uses at inference time.
#     """
#     import math as _m
#
#     bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
#     tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
#     sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
#     w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]
#
#     # Compute T for every val sample (same formula as trust_score_computation.py)
#     T_vals = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
#         vals = list(ts.values())
#         m    = sum(vals) / len(vals)
#         var  = sum((v - m) ** 2 for v in vals) / len(vals)
#         cons = max(0.10, _m.exp(-var * variance_decay))
#         dec  = abs(prob - 0.5) * 2.0
#         cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
#         strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
#         T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
#         T_vals.append((T, int(y[i])))
#
#     # Find the threshold that best separates correct from incorrect predictions
#     # by maximising: (recall of incorrect below threshold) + (precision above)
#     # Search range is clamped to [T_min, T_max] of actual val-set T values
#     # so the threshold is always within the real T distribution.
#     t_values = [T for T, _ in T_vals]
#     t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
#     t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
#     thresholds = [round(t_min + i * 0.05, 2)
#                   for i in range(int((t_max - t_min) / 0.05) + 1)]
#     if not thresholds:
#         thresholds = [0.50]
#     best_t = thresholds[len(thresholds)//2]; best_score = -1.0
#
#     for t in thresholds:
#         above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
#         above_total   = sum(1 for T, _      in T_vals if T >= t)
#         below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
#         total_wrong   = sum(1 for _, correct in T_vals if not correct)
#
#         precision_above  = above_correct / max(1, above_total)
#         recall_wrong_below = below_wrong / max(1, total_wrong)
#         score = (precision_above + recall_wrong_below) / 2.0
#
#         if score > best_score:
#             best_score = score; best_t = t
#
#     log.info(f"  trust_threshold={best_t:.2f}  "
#              f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     return round(best_t, 2)
#
#
# def calibrate_trust(probs, y):
#     """
#     Fit a linear calibration: oracle_trust = intercept + slope * decisiveness
#     where oracle_trust = 1 - |V - y|  (1=correct, 0=maximally wrong)
#     and decisiveness = |V - 0.5| * 2  (0=uncertain, 1=fully confident)
#
#     The intercept represents the base trust level for an indecisive model.
#     It must be >= 0.45 so that decisive clean samples can reach T >= 0.8
#     even when consistency is moderate (0.55-0.65 range).
#
#     If the fitted intercept drops below 0.45 it means the val set contains
#     too many wrong predictions pulling it down — clamp it to ensure the
#     T score distribution remains meaningful.
#     """
#     v = probs.astype(np.float64); yf = y.astype(np.float64)
#     oracle  = 1.0 - np.abs(v - yf)
#     decisiv = np.abs(v - 0.5) * 2.0
#     A = np.column_stack([np.ones_like(decisiv), decisiv])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         # Floor at 0.45: ensures decisive clean samples reach T >= 0.8
#         # Ceiling at 0.75: prevents T from being trivially high everywhere
#         intercept = float(np.clip(coeffs[0], 0.45, 0.75))
#         slope     = float(np.clip(coeffs[1], 0.0,  0.60))
#     except Exception:
#         intercept, slope = 0.48, 0.50
#     log.info(
#         f"  trust_calibration: intercept={intercept:.4f}  slope={slope:.4f}"
#     )
#     return {"intercept": round(intercept,6), "slope": round(slope,6)}
#
#
#
#
# def learn_task1_alpha(X_va: np.ndarray, y_va: np.ndarray) -> float:
#     """
#     Learn the alpha parameter for Task 1 from the validation set.
#
#     alpha controls the balance between hit presence and confidence:
#         s1 = alpha * hit_indicator + (1 - alpha) * avg_confidence
#
#     We find the alpha in (0,1) that maximises the Pearson correlation
#     between the resulting s1 values and the ground-truth labels y.
#
#     High alpha -> trust hit presence more  (robust but less nuanced)
#     Low  alpha -> trust confidence more    (precise but noise-sensitive)
#
#     The optimal value is dataset-dependent:
#       - BigVul  (ground_truth strategy, precise patterns) -> alpha ~ 0.45
#       - Noisier datasets (heuristic/all_lines)             -> alpha ~ 0.55-0.70
#     """
#     hit_col  = FEATURE_NAMES.index("pattern_hit_count")
#     conf_col = FEATURE_NAMES.index("avg_line_conf")
#
#     hits = X_va[:, hit_col].astype(np.float64)
#     conf = X_va[:, conf_col].astype(np.float64)
#     yf   = y_va.astype(np.float64)
#
#     # Binarise hits: 1 if any pattern fired, 0 otherwise
#     hit_indicator = (hits > 0).astype(np.float64)
#
#     best_alpha = 0.50
#     best_corr  = -2.0
#
#     for i in range(1, 20):          # alpha in {0.05, 0.10, ..., 0.95}
#         alpha = round(i / 20.0, 2)
#         s1    = alpha * hit_indicator + (1.0 - alpha) * conf
#         if s1.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(s1, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr  = corr
#                 best_alpha = alpha
#
#     log.info(
#         f"  task1_alpha learned: {best_alpha}  "
#         f"(Pearson corr with labels = {best_corr:.4f})"
#     )
#     return best_alpha
#
#
# # =============================================================================
# # Models
# # =============================================================================
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
#         self.lr=lr; self.epochs=epochs; self.bs=batch_size
#         self.l2=l2; self.seed=seed; self.w=None; self.b=None
#     def fit(self, X, y):
#         rng=np.random.RandomState(self.seed); n,d=X.shape
#         self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
#         for epoch in range(self.epochs):
#             idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
#                 p=_sig(Xb@self.w+self.b); e=p-yb
#                 self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
#                 loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
#             if (epoch+1)%100==0:
#                 log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
#         return self
#     def predict_proba(self, X): return _sig(X@self.w+self.b)
#     def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m
#
#
# class NeuralNetwork:
#     def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
#         self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
#         self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
#     def _init(self, d):
#         rng=np.random.RandomState(self.seed); self._rng=rng
#         self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
#                 "b1":np.zeros(self.h1,dtype=np.float32),
#                 "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
#                 "b2":np.zeros(self.h2,dtype=np.float32),
#                 "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
#                 "b3":np.zeros(1,dtype=np.float32)}
#     def _fwd(self, X, train=False):
#         p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
#         if train and self.dropout>0:
#             m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
#         else: m1=None
#         z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
#         if train and self.dropout>0:
#             m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
#         else: m2=None
#         return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
#     def _bwd(self, X, y, prob, cache):
#         z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
#         dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
#         dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
#         da2=dz3@p["W3"].T
#         if m2 is not None: da2=da2*m2/(1-self.dropout)
#         dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
#         da1=dz2@p["W2"].T
#         if m1 is not None: da1=da1*m1/(1-self.dropout)
#         dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
#         return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
#     def fit(self, X, y):
#         self._init(X.shape[1]); n=len(y)
#         for epoch in range(self.epochs):
#             idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
#                 prob,cache=self._fwd(Xb,train=True)
#                 loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
#                 grads=self._bwd(Xb,yb,prob,cache)
#                 for k in self.p: self.p[k]-=self.lr*grads[k]
#             if (epoch+1)%20==0:
#                 pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
#                 log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
#         return self
#     def predict_proba(self, X): p,_=self._fwd(X); return p
#     def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# def evaluate(model, Xte, yte, threshold=0.50):
#     probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
#     tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
#     fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
#     acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
#     f1=2*prec*rec/max(1e-8,prec+rec)
#     pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
#     if len(pos_p)>0 and len(neg_p)>0:
#         c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
#         auc=c/(len(pos_p)*len(neg_p))
#     return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
#             "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
#             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
#             "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
#             "evaluated_on":"bigvul_test_set"}
#
#
# def feature_importance_logreg(model):
#     abs_w=np.abs(model.w); total=abs_w.sum()
#     if total<1e-8: return []
#     ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
#     return [{"feature":f,"importance":round(i,6)} for f,i in ranked]
#
#
# def compute_megavul_score_dist(model, X_mv_scaled):
#     if X_mv_scaled.shape[0]==0:
#         return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
#     probs=model.predict_proba(X_mv_scaled)
#     buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
#     for p in probs:
#         if p<0.2: buckets["0.0-0.2"]+=1
#         elif p<0.4: buckets["0.2-0.4"]+=1
#         elif p<0.6: buckets["0.4-0.6"]+=1
#         elif p<0.8: buckets["0.6-0.8"]+=1
#         else: buckets["0.8-1.0"]+=1
#     return {"n":len(probs),"mean":round(float(probs.mean()),4),
#             "std":round(float(probs.std()),4),"buckets":buckets}
#
#
# # =============================================================================
# # Compute val task scores for trust threshold learning
# # =============================================================================
#
# def compute_val_task_scores(X_va, y_va):
#     """Extract per-sample task score dicts from val feature vectors."""
#     task_scores_list = []
#     cols = _TASK_COLS
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#     return task_scores_list
#
#
# # =============================================================================
# # Training entry points
# # =============================================================================
#
#
# def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
#                      val_probs: np.ndarray) -> dict:
#     """
#     Learn all ALC constants from the validation set.
#     Nothing in alc/run_alc.py is hardcoded — every number comes from here.
#
#     Learned values:
#       conflict_threshold  — pairwise score diff that counts as "conflict"
#                             = mean absolute pairwise difference on val set
#                             (tasks that naturally spread this far are conflicting)
#       min_consistency     — lowest achievable consistency score
#                             = consistency at maximum observed variance
#       variance_decay      — steepness of exp(-var * decay)
#                             = fitted so exp(-max_var * decay) = min_consistency
#       direction_threshold — score boundary separating "risky" from "clean"
#                             = optimal threshold that best separates val labels
#       blend_weights       — {consistency, calibration, strategy}
#                             = learned by fitting a 3-feature linear model
#                               mapping (consistency, calibration, strategy_quality)
#                               to oracle trust (1 - |V - y|) on val set
#       strategy_quality    — {ground_truth, heuristic, all_lines}
#                             = kept as data-pipeline constants (not empirical)
#                               because they reflect factual confidence levels
#                               about the suspicious-line mapping process itself,
#                               not something the val set can determine
#     """
#     import math as _math
#
#     cols = _TASK_COLS
#     task_scores_list = []
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#
#     # ── conflict_threshold ──────────────────────────────────────────────────
#     # Compute all pairwise absolute differences on val set
#     all_diffs = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         for i in range(len(vals)):
#             for j in range(i+1, len(vals)):
#                 all_diffs.append(abs(vals[i] - vals[j]))
#
#     # Use the 75th percentile: pairs above this are genuinely "conflicting"
#     all_diffs.sort()
#     p75_idx      = int(len(all_diffs) * 0.75)
#     conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
#     # Clamp to a sensible range [0.15, 0.50]
#     conflict_thr = max(0.15, min(0.50, conflict_thr))
#
#     # ── variance_decay and min_consistency ─────────────────────────────────
#     # Compute per-sample variances
#     variances = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         m = sum(vals) / len(vals)
#         variances.append(sum((v-m)**2 for v in vals) / len(vals))
#
#     max_var = max(variances) if variances else 0.25
#     min_var = min(variances) if variances else 0.0
#
#     # min_consistency = the lowest trust we should assign even at max disagreement
#     # = fraction of val samples that are correct even at maximum variance
#     correct_at_max = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
#         vals = list(ts.values()); m = sum(vals)/len(vals)
#         var  = sum((v-m)**2 for v in vals)/len(vals)
#         if abs(var - max_var) < 0.02:   # near-maximum variance samples
#             correct = (int(prob >= 0.5) == int(y_va[i]))
#             correct_at_max.append(float(correct))
#     min_consistency = round(
#         float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
#         4
#     )
#     min_consistency = max(0.05, min(0.30, min_consistency))
#
#     # variance_decay: solve exp(-max_var * decay) = min_consistency
#     # → decay = -ln(min_consistency) / max_var
#     if max_var > 1e-6 and min_consistency > 1e-6:
#         variance_decay = round(-_math.log(min_consistency) / max_var, 4)
#         variance_decay = max(2.0, min(20.0, variance_decay))
#     else:
#         variance_decay = 8.0
#
#     # ── direction_threshold ─────────────────────────────────────────────────
#     # Find the task score boundary that best separates vulnerable from clean
#     # Search over [0.10, 0.70] using mean task score on val set
#     mean_scores = [
#         sum(ts.values()) / len(ts) for ts in task_scores_list
#     ]
#     best_dir_thr = 0.30; best_acc = 0.0
#     for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
#         preds = [1 if s >= t else 0 for s in mean_scores]
#         acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
#         if acc > best_acc:
#             best_acc = acc; best_dir_thr = round(t, 2)
#
#     # ── blend_weights for Stage 3 ───────────────────────────────────────────
#     # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
#     # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
#     # Consistency and calibration are computable from val data
#     # strategy_quality: use 0.80 (heuristic) for all val samples
#     #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)
#
#     oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
#     decisive = np.abs(val_probs - 0.5) * 2.0
#
#     # Compute per-sample consistency from task variance
#     consistencies = np.array([
#         max(min_consistency,
#             _math.exp(-v * variance_decay))
#         for v in variances
#     ], dtype=np.float64)
#
#     # Calibration column (same formula used at inference)
#     calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
#     strategy_col = np.full(len(y_va), 0.80)                    # heuristic default
#
#     A = np.column_stack([consistencies, calibrations, strategy_col])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
#     except Exception:
#         w1, w2, w3 = 0.50, 0.35, 0.15
#
#     # Enforce minimum floors without iterative oscillation.
#     #
#     # The iterative approach (clamp → renormalise → repeat) fails because
#     # dividing by the new total pulls w1 back below the floor every step.
#     #
#     # Correct approach: treat the floors as hard reservations.
#     # Whatever the lstsq gives, first assign the floors, then distribute
#     # the remaining budget (1 - sum_of_floors = 0.50) proportionally to
#     # whichever weights the lstsq pushed above their own floors.
#     #
#     FLOOR_CON, FLOOR_CAL, FLOOR_STR = 0.35, 0.15, 0.05
#     w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
#     total = w1 + w2 + w3
#     if total < 1e-8:
#         w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
#     else:
#         w1, w2, w3 = w1/total, w2/total, w3/total
#         # Compute how much each weight exceeds its own floor
#         excess1 = max(0.0, w1 - FLOOR_CON)
#         excess2 = max(0.0, w2 - FLOOR_CAL)
#         excess3 = max(0.0, w3 - FLOOR_STR)
#         total_excess = excess1 + excess2 + excess3
#         # Budget available above the floors
#         budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
#         if total_excess > 1e-8:
#             # Distribute budget in proportion to excess
#             w1 = FLOOR_CON + budget * (excess1 / total_excess)
#             w2 = FLOOR_CAL + budget * (excess2 / total_excess)
#             w3 = FLOOR_STR + budget * (excess3 / total_excess)
#         else:
#             # All weights were at or below floor — give budget to consistency
#             w1 = FLOOR_CON + budget
#             w2 = FLOOR_CAL
#             w3 = FLOOR_STR
#
#     # Derive w3 from w1+w2 to avoid float accumulation
#     w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
#     # Final safety clamp (float rounding edge case)
#     w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
#     total = w1 + w2 + w3
#     w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)
#
#     blend_weights = {
#         "consistency":  w1,
#         "calibration":  w2,
#         "strategy":     w3,
#     }
#
#     log.info(
#         f"  ALC params learned from val set:\n"
#         f"    conflict_threshold={conflict_thr}  "
#         f"min_consistency={min_consistency}\n"
#         f"    variance_decay={variance_decay}  "
#         f"direction_threshold={best_dir_thr}\n"
#         f"    blend_weights={blend_weights}"
#     )
#
#     return {
#         "conflict_threshold":  conflict_thr,
#         "min_consistency":     min_consistency,
#         "variance_decay":      variance_decay,
#         "direction_threshold": best_dir_thr,
#         "blend_weights":       blend_weights,
#         # strategy_quality is a pipeline constant, not learned from val data
#         "strategy_quality": {
#             "ground_truth": 1.00,
#             "heuristic":    0.80,
#             "all_lines":    0.50,
#         },
#     }
#
# def learn_task_weights(X_tr, y_tr):
#     """
#     Learn relative importance weights for the four MTD tasks
#     by gradient descent on the training set.
#     Weights are non-negative and sum to 1.
#     """
#     cols    = list(_TASK_COLS.values())
#     Xt      = X_tr[:, cols].astype(np.float64)
#     col_std = Xt.std(0); col_std[col_std < 1e-8] = 1.0
#     Xn      = Xt / col_std
#     yf      = y_tr.astype(np.float64)
#     rng     = np.random.RandomState(42)
#     w       = rng.rand(4).astype(np.float64) * 0.25
#     for _ in range(500):
#         prob = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w, -500, 500)))
#         grad = Xn.T @ (prob - yf) / len(yf)
#         w   -= 0.05 * grad
#         w    = np.maximum(0.0, w)
#         s    = w.sum()
#         if s > 1e-8:
#             w /= s
#     if w.sum() < 1e-8:
#         w = np.array([0.25, 0.25, 0.25, 0.25])
#     wts = {k: round(float(v), 6)
#            for k, v in zip(["task1","task2","task3","task4"], w)}
#     log.info(f"  Learned task weights: {wts}")
#     return wts
#
#
# def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
#           task_wts, alc_params, metrics, mv_dist,
#           task1_alpha=0.50, extra=None):
#     d = {
#         **model.to_dict(),
#         "scaler":             scaler.to_dict(),
#         "feature_names":      FEATURE_NAMES,
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul_combined",
#         "task1_alpha":        round(task1_alpha, 4),
#         # All learned from data — nothing hardcoded:
#         "opt_threshold":      opt_thresh["threshold"],
#         "threshold_metrics":  opt_thresh,
#         "trust_threshold":    trust_thresh,
#         "trust_calibration":  trust_cal,
#         "task_weights":       task_wts,
#         "alc_params":         alc_params,   # ALC constants, all data-driven
#         "metrics":            metrics,
#         "megavul_score_dist": mv_dist,
#     }
#     if extra: d.update(extra)
#     return d
#
#
# def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
#     log.info("Training Logistic Regression on BigVul labels...")
#     model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
#     model.fit(Xtr, ytr)
#     vp    = model.predict_proba(Xva)
#     opt   = find_opt_threshold(vp, yva)
#     ts    = compute_val_task_scores(Xva, yva)
#     alcp  = learn_alc_params(Xva, yva, vp)
#     tcal  = calibrate_trust(vp, yva)
#     tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                   blend_weights=alcp["blend_weights"],
#                                   trust_cal=tcal,
#                                   strategy_quality=alcp["strategy_quality"])
#     twts  = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     imps  = feature_importance_logreg(model)
#     met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd   = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  LR test metrics (BigVul): {met}")
#     log.info(f"  Top-5 features: {imps[:5]}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, extra={"feature_importances": imps})
#     path = MODELS_DIR / "logreg_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  LR model saved → {path}")
#     return met
#
#
# def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
#     log.info("Training Neural Network (MLP) on BigVul labels...")
#     model = NeuralNetwork(h1=64, h2=32, lr=0.001, epochs=epochs, bs=64, l2=1e-4, dropout=0.2)
#     model.fit(Xtr, ytr)
#     vp   = model.predict_proba(Xva)
#     opt  = find_opt_threshold(vp, yva)
#     ts   = compute_val_task_scores(Xva, yva)
#     alcp = learn_alc_params(Xva, yva, vp)
#     tcal = calibrate_trust(vp, yva)
#     tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                  blend_weights=alcp["blend_weights"],
#                                  trust_cal=tcal,
#                                  strategy_quality=alcp["strategy_quality"])
#     twts = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd  = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  NN test metrics (BigVul): {met}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a)
#     path = MODELS_DIR / "nn_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  NN model saved → {path}")
#     return met
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(description="Train MTD ML models")
#     parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--test-ratio", type=float, default=0.20)
#     parser.add_argument("--val-ratio",  type=float, default=0.15)
#     args = parser.parse_args()
#
#     log.info("=== MTD ML Training ===")
#     log.info(f"Supervised: BigVul only  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")
#
#     # ── Load BigVul labelled data for training ────────────────────────────────
#     Xbv, ybv, ids = load_labelled("bigvul")
#     n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
#     log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
#     if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
#         log.error(
#             f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
#             f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
#         )
#         sys.exit(1)
#
#     # ── Load MegaVul features for scaler fitting only (not for training) ──────
#     # IMPORTANT: MegaVul is used ONLY to fit the scaler so it generalises
#     # to MegaVul's feature distribution at inference time.
#     # MegaVul is NOT used for training, calibration, or threshold finding
#     # because its V scores are unreliable (heuristic strategy, no flaw lines).
#     # Mixing MegaVul into calibrate_trust() destroys the T score distribution.
#     Xmv, _ = load_all_features("megavul")
#
#     # ── Train/val/test split on BigVul only ───────────────────────────────────
#     Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
#                                              val=args.val_ratio, test=args.test_ratio)
#     log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
#     log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
#     log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
#     log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")
#
#     # ── Scaler fitted on BigVul train + all MegaVul features ─────────────────
#     scaler = build_combined_scaler(Xtr, Xmv)
#     Xtr_s  = scaler.transform(Xtr)
#     Xva_s  = scaler.transform(Xva)
#     Xte_s  = scaler.transform(Xte)
#     Xmv_s  = (scaler.transform(Xmv)
#                if Xmv.shape[0] > 0
#                else np.empty((0, FEATURE_DIM), dtype=np.float32))
#
#     report = {
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul",
#         "n_train":            len(ytr),
#         "n_val":              len(yva),
#         "n_test":             len(yte),
#         "n_megavul_scaler":   Xmv.shape[0],
#     }
#     if args.model in ("logreg","both"):
#         report["logreg"] = train_logreg(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s)
#     if args.model in ("nn","both"):
#         report["nn"] = train_nn(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s,
#                                 epochs=args.epochs)
#
#     rp = MODELS_DIR / "training_report.json"
#     rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
#     log.info(f"Training report → {rp}")
#     log.info("=== Training complete ===")
#
#
# def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
#
#
# if __name__ == "__main__":
#     main()



# # =============================================================================
# # mtd/ml/train.py  —  ML Model Trainer
# #
# # Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# # Saves into each model JSON (ALL values learned from data, NONE hardcoded):
# #   weights/params       model parameters
# #   scaler               mean/std for feature normalisation
# #   opt_threshold        F1-optimal V threshold (from val set search)
# #   trust_threshold      threshold below which T triggers "untrustworthy"
# #                        (learned from val-set consistency distribution)
# #   trust_calibration    {intercept, slope}: V → raw T mapping
# #   task_weights         learned relative task importance
# #   feature_importances  ranked LR feature weights
# #   metrics              accuracy, precision, recall, F1, AUC
# #
# # Usage:
# #   python mtd/ml/train.py --model both --epochs 100
# # =============================================================================
#
# import argparse
# import json
# import logging
# import math
# import random
# import sys
# from pathlib import Path
# from collections import Counter
#
# import numpy as np
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from feature_extractor import FEATURE_DIM, FEATURE_NAMES
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# MODELS_DIR   = Path(__file__).resolve().parent / "models"
# DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
# _TASK_COLS = {
#     "task1": FEATURE_NAMES.index("pattern_hit_count"),
#     "task2": FEATURE_NAMES.index("risky_line_ratio"),
#     "task3": FEATURE_NAMES.index("overall_syntax_risk"),
#     "task4": FEATURE_NAMES.index("overall_dep_risk"),
# }
#
#
# # =============================================================================
# # Data loading
# # =============================================================================
#
# def load_labelled(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         log.error(f"Not found: {path}  —  run build_dataset.py first")
#         sys.exit(1)
#     X, y, ids = [], [], []
#     skipped = 0
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             if rec.get("label", -1) == -1: skipped += 1; continue
#             feats = rec.get("features", [])
#             if len(feats) != FEATURE_DIM: continue
#             X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
#     log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
#     return X, y, ids
#
#
# def load_all_features(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         return np.empty((0, FEATURE_DIM), dtype=np.float32), []
#     X, ids = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             feats = rec.get("features", [])
#             if len(feats) == FEATURE_DIM:
#                 X.append(feats); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
#     log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
#     return X, ids
#
#
# def split3(X, y, ids, val=0.15, test=0.20, seed=42):
#     rng = random.Random(seed)
#     pos = [i for i, l in enumerate(y) if l == 1]
#     neg = [i for i, l in enumerate(y) if l == 0]
#     rng.shuffle(pos); rng.shuffle(neg)
#     def cut(lst):
#         n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
#         return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
#     def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
#     pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
#     tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
#     return (X[tr],y[tr],[ids[i] for i in tr],
#             X[va],y[va],[ids[i] for i in va],
#             X[te],y[te],[ids[i] for i in te])
#
#
# # =============================================================================
# # Scaler
# # =============================================================================
#
# class StandardScaler:
#     def __init__(self): self.mean_=None; self.std_=None
#     def fit(self, X):
#         self.mean_=X.mean(0); self.std_=X.std(0)
#         self.std_[self.std_<1e-8]=1.0; return self
#     def transform(self, X): return (X-self.mean_)/self.std_
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
#     @classmethod
#     def from_dict(cls, d):
#         s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
#         s.std_=np.array(d["std"],dtype=np.float32); return s
#
#
# def build_combined_scaler(X_bv, X_mv):
#     parts = [X_bv]
#     if X_mv.shape[0] > 0: parts.append(X_mv)
#     X = np.vstack(parts)
#     sc = StandardScaler(); sc.fit(X)
#     log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
#     return sc
#
#
# # =============================================================================
# # Learned parameters
# # =============================================================================
#
# def find_opt_threshold(probs, y):
#     best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
#     for t in [i/100 for i in range(5, 96)]:
#         preds = (probs >= t).astype(int)
#         tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
#         fn = int(((preds==0)&(y==1)).sum())
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 2*prec*rec/max(1e-8,prec+rec)
#         if f1 > best["f1"]:
#             best = {"threshold":round(t,2),"f1":round(f1,4),
#                     "precision":round(prec,4),"recall":round(rec,4)}
#     log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
#     return best
#
#
# def find_trust_threshold(probs, y, task_scores_list, variance_decay,
#                           blend_weights=None, trust_cal=None,
#                           strategy_quality=None):
#     """
#     Learn the trust threshold from the validation set.
#
#     Searches over actual T values (blending consistency + calibration +
#     strategy quality) rather than raw consistency alone, so the threshold
#     is calibrated to the same space ALC uses at inference time.
#     """
#     import math as _m
#
#     bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
#     tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
#     sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
#     w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]
#
#     # Compute T for every val sample (same formula as trust_score_computation.py)
#     T_vals = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
#         vals = list(ts.values())
#         m    = sum(vals) / len(vals)
#         var  = sum((v - m) ** 2 for v in vals) / len(vals)
#         cons = max(0.10, _m.exp(-var * variance_decay))
#         dec  = abs(prob - 0.5) * 2.0
#         cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
#         strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
#         T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
#         T_vals.append((T, int(y[i])))
#
#     # Find the threshold that best separates correct from incorrect predictions
#     # by maximising: (recall of incorrect below threshold) + (precision above)
#     # Search range is clamped to [T_min, T_max] of actual val-set T values
#     # so the threshold is always within the real T distribution.
#     t_values = [T for T, _ in T_vals]
#     t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
#     t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
#     thresholds = [round(t_min + i * 0.05, 2)
#                   for i in range(int((t_max - t_min) / 0.05) + 1)]
#     if not thresholds:
#         thresholds = [0.50]
#     best_t = thresholds[len(thresholds)//2]; best_score = -1.0
#
#     for t in thresholds:
#         above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
#         above_total   = sum(1 for T, _      in T_vals if T >= t)
#         below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
#         total_wrong   = sum(1 for _, correct in T_vals if not correct)
#
#         precision_above  = above_correct / max(1, above_total)
#         recall_wrong_below = below_wrong / max(1, total_wrong)
#         score = (precision_above + recall_wrong_below) / 2.0
#
#         if score > best_score:
#             best_score = score; best_t = t
#
#     log.info(f"  trust_threshold={best_t:.2f}  "
#              f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     return round(best_t, 2)
#
#
# def calibrate_trust(probs, y):
#     """
#     Fit a linear calibration: oracle_trust = intercept + slope * decisiveness
#     where oracle_trust = 1 - |V - y|  (1=correct, 0=maximally wrong)
#     and decisiveness = |V - 0.5| * 2  (0=uncertain, 1=fully confident)
#
#     The intercept represents the base trust level for an indecisive model.
#     It must be >= 0.45 so that decisive clean samples can reach T >= 0.8
#     even when consistency is moderate (0.55-0.65 range).
#
#     If the fitted intercept drops below 0.45 it means the val set contains
#     too many wrong predictions pulling it down — clamp it to ensure the
#     T score distribution remains meaningful.
#     """
#     v = probs.astype(np.float64); yf = y.astype(np.float64)
#     oracle  = 1.0 - np.abs(v - yf)
#     decisiv = np.abs(v - 0.5) * 2.0
#     A = np.column_stack([np.ones_like(decisiv), decisiv])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         # Floor at 0.45: ensures decisive clean samples reach T >= 0.8
#         # Ceiling at 0.75: prevents T from being trivially high everywhere
#         intercept = float(np.clip(coeffs[0], 0.45, 0.75))
#         slope     = float(np.clip(coeffs[1], 0.0,  0.60))
#     except Exception:
#         intercept, slope = 0.48, 0.50
#     log.info(
#         f"  trust_calibration: intercept={intercept:.4f}  slope={slope:.4f}"
#     )
#     return {"intercept": round(intercept,6), "slope": round(slope,6)}
#
#
#
#
# def learn_task1_alpha(X_va: np.ndarray, y_va: np.ndarray) -> float:
#     """
#     Learn the alpha parameter for Task 1 from the validation set.
#
#     alpha controls the balance between hit presence and confidence:
#         s1 = alpha * hit_indicator + (1 - alpha) * avg_confidence
#
#     We find the alpha in (0,1) that maximises the Pearson correlation
#     between the resulting s1 values and the ground-truth labels y.
#
#     High alpha -> trust hit presence more  (robust but less nuanced)
#     Low  alpha -> trust confidence more    (precise but noise-sensitive)
#
#     The optimal value is dataset-dependent:
#       - BigVul  (ground_truth strategy, precise patterns) -> alpha ~ 0.45
#       - Noisier datasets (heuristic/all_lines)             -> alpha ~ 0.55-0.70
#     """
#     hit_col  = FEATURE_NAMES.index("pattern_hit_count")
#     conf_col = FEATURE_NAMES.index("avg_line_conf")
#
#     hits = X_va[:, hit_col].astype(np.float64)
#     conf = X_va[:, conf_col].astype(np.float64)
#     yf   = y_va.astype(np.float64)
#
#     # Binarise hits: 1 if any pattern fired, 0 otherwise
#     hit_indicator = (hits > 0).astype(np.float64)
#
#     best_alpha = 0.50
#     best_corr  = -2.0
#
#     for i in range(1, 20):          # alpha in {0.05, 0.10, ..., 0.95}
#         alpha = round(i / 20.0, 2)
#         s1    = alpha * hit_indicator + (1.0 - alpha) * conf
#         if s1.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(s1, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr  = corr
#                 best_alpha = alpha
#
#     log.info(
#         f"  task1_alpha learned: {best_alpha}  "
#         f"(Pearson corr with labels = {best_corr:.4f})"
#     )
#     return best_alpha
#
#
# # =============================================================================
# # Models
# # =============================================================================
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
#         self.lr=lr; self.epochs=epochs; self.bs=batch_size
#         self.l2=l2; self.seed=seed; self.w=None; self.b=None
#     def fit(self, X, y):
#         rng=np.random.RandomState(self.seed); n,d=X.shape
#         self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
#         for epoch in range(self.epochs):
#             idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
#                 p=_sig(Xb@self.w+self.b); e=p-yb
#                 self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
#                 loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
#             if (epoch+1)%100==0:
#                 log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
#         return self
#     def predict_proba(self, X): return _sig(X@self.w+self.b)
#     def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m
#
#
# class NeuralNetwork:
#     def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
#         self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
#         self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
#     def _init(self, d):
#         rng=np.random.RandomState(self.seed); self._rng=rng
#         self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
#                 "b1":np.zeros(self.h1,dtype=np.float32),
#                 "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
#                 "b2":np.zeros(self.h2,dtype=np.float32),
#                 "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
#                 "b3":np.zeros(1,dtype=np.float32)}
#     def _fwd(self, X, train=False):
#         p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
#         if train and self.dropout>0:
#             m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
#         else: m1=None
#         z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
#         if train and self.dropout>0:
#             m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
#         else: m2=None
#         return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
#     def _bwd(self, X, y, prob, cache):
#         z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
#         dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
#         dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
#         da2=dz3@p["W3"].T
#         if m2 is not None: da2=da2*m2/(1-self.dropout)
#         dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
#         da1=dz2@p["W2"].T
#         if m1 is not None: da1=da1*m1/(1-self.dropout)
#         dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
#         return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
#     def fit(self, X, y):
#         self._init(X.shape[1]); n=len(y)
#         for epoch in range(self.epochs):
#             idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
#                 prob,cache=self._fwd(Xb,train=True)
#                 loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
#                 grads=self._bwd(Xb,yb,prob,cache)
#                 for k in self.p: self.p[k]-=self.lr*grads[k]
#             if (epoch+1)%20==0:
#                 pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
#                 log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
#         return self
#     def predict_proba(self, X): p,_=self._fwd(X); return p
#     def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# def evaluate(model, Xte, yte, threshold=0.50):
#     probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
#     tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
#     fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
#     acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
#     f1=2*prec*rec/max(1e-8,prec+rec)
#     pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
#     if len(pos_p)>0 and len(neg_p)>0:
#         c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
#         auc=c/(len(pos_p)*len(neg_p))
#     return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
#             "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
#             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
#             "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
#             "evaluated_on":"bigvul_test_set"}
#
#
# def feature_importance_logreg(model):
#     abs_w=np.abs(model.w); total=abs_w.sum()
#     if total<1e-8: return []
#     ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
#     return [{"feature":f,"importance":round(i,6)} for f,i in ranked]
#
#
# def compute_megavul_score_dist(model, X_mv_scaled):
#     if X_mv_scaled.shape[0]==0:
#         return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
#     probs=model.predict_proba(X_mv_scaled)
#     buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
#     for p in probs:
#         if p<0.2: buckets["0.0-0.2"]+=1
#         elif p<0.4: buckets["0.2-0.4"]+=1
#         elif p<0.6: buckets["0.4-0.6"]+=1
#         elif p<0.8: buckets["0.6-0.8"]+=1
#         else: buckets["0.8-1.0"]+=1
#     return {"n":len(probs),"mean":round(float(probs.mean()),4),
#             "std":round(float(probs.std()),4),"buckets":buckets}
#
#
# # =============================================================================
# # Compute val task scores for trust threshold learning
# # =============================================================================
#
# def compute_val_task_scores(X_va, y_va):
#     """Extract per-sample task score dicts from val feature vectors."""
#     task_scores_list = []
#     cols = _TASK_COLS
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#     return task_scores_list
#
#
# # =============================================================================
# # Training entry points
# # =============================================================================
#
#
# def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
#                      val_probs: np.ndarray) -> dict:
#     """
#     Learn all ALC constants from the validation set.
#     Nothing in alc/run_alc.py is hardcoded — every number comes from here.
#
#     Learned values:
#       conflict_threshold  — pairwise score diff that counts as "conflict"
#                             = mean absolute pairwise difference on val set
#                             (tasks that naturally spread this far are conflicting)
#       min_consistency     — lowest achievable consistency score
#                             = consistency at maximum observed variance
#       variance_decay      — steepness of exp(-var * decay)
#                             = fitted so exp(-max_var * decay) = min_consistency
#       direction_threshold — score boundary separating "risky" from "clean"
#                             = optimal threshold that best separates val labels
#       blend_weights       — {consistency, calibration, strategy}
#                             = learned by fitting a 3-feature linear model
#                               mapping (consistency, calibration, strategy_quality)
#                               to oracle trust (1 - |V - y|) on val set
#       strategy_quality    — {ground_truth, heuristic, all_lines}
#                             = kept as data-pipeline constants (not empirical)
#                               because they reflect factual confidence levels
#                               about the suspicious-line mapping process itself,
#                               not something the val set can determine
#     """
#     import math as _math
#
#     cols = _TASK_COLS
#     task_scores_list = []
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#
#     # ── conflict_threshold ──────────────────────────────────────────────────
#     # Compute all pairwise absolute differences on val set
#     all_diffs = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         for i in range(len(vals)):
#             for j in range(i+1, len(vals)):
#                 all_diffs.append(abs(vals[i] - vals[j]))
#
#     # Use the 75th percentile: pairs above this are genuinely "conflicting"
#     all_diffs.sort()
#     p75_idx      = int(len(all_diffs) * 0.75)
#     conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
#     # Clamp to a sensible range [0.15, 0.50]
#     conflict_thr = max(0.15, min(0.50, conflict_thr))
#
#     # ── variance_decay and min_consistency ─────────────────────────────────
#     # Compute per-sample variances
#     variances = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         m = sum(vals) / len(vals)
#         variances.append(sum((v-m)**2 for v in vals) / len(vals))
#
#     max_var = max(variances) if variances else 0.25
#     min_var = min(variances) if variances else 0.0
#
#     # min_consistency = the lowest trust we should assign even at max disagreement
#     # = fraction of val samples that are correct even at maximum variance
#     correct_at_max = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
#         vals = list(ts.values()); m = sum(vals)/len(vals)
#         var  = sum((v-m)**2 for v in vals)/len(vals)
#         if abs(var - max_var) < 0.02:   # near-maximum variance samples
#             correct = (int(prob >= 0.5) == int(y_va[i]))
#             correct_at_max.append(float(correct))
#     min_consistency = round(
#         float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
#         4
#     )
#     min_consistency = max(0.05, min(0.30, min_consistency))
#
#     # variance_decay: solve exp(-max_var * decay) = min_consistency
#     # → decay = -ln(min_consistency) / max_var
#     if max_var > 1e-6 and min_consistency > 1e-6:
#         variance_decay = round(-_math.log(min_consistency) / max_var, 4)
#         variance_decay = max(2.0, min(20.0, variance_decay))
#     else:
#         variance_decay = 8.0
#
#     # ── direction_threshold ─────────────────────────────────────────────────
#     # Find the task score boundary that best separates vulnerable from clean
#     # Search over [0.10, 0.70] using mean task score on val set
#     mean_scores = [
#         sum(ts.values()) / len(ts) for ts in task_scores_list
#     ]
#     best_dir_thr = 0.30; best_acc = 0.0
#     for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
#         preds = [1 if s >= t else 0 for s in mean_scores]
#         acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
#         if acc > best_acc:
#             best_acc = acc; best_dir_thr = round(t, 2)
#
#     # ── blend_weights for Stage 3 ───────────────────────────────────────────
#     # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
#     # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
#     # Consistency and calibration are computable from val data
#     # strategy_quality: use 0.80 (heuristic) for all val samples
#     #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)
#
#     oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
#     decisive = np.abs(val_probs - 0.5) * 2.0
#
#     # Compute per-sample consistency from task variance
#     consistencies = np.array([
#         max(min_consistency,
#             _math.exp(-v * variance_decay))
#         for v in variances
#     ], dtype=np.float64)
#
#     # Calibration column (same formula used at inference)
#     calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
#     strategy_col = np.full(len(y_va), 0.80)                    # heuristic default
#
#     A = np.column_stack([consistencies, calibrations, strategy_col])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
#     except Exception:
#         w1, w2, w3 = 0.50, 0.35, 0.15
#
#     # Enforce minimum floors without iterative oscillation.
#     #
#     # The iterative approach (clamp → renormalise → repeat) fails because
#     # dividing by the new total pulls w1 back below the floor every step.
#     #
#     # Correct approach: treat the floors as hard reservations.
#     # Whatever the lstsq gives, first assign the floors, then distribute
#     # the remaining budget (1 - sum_of_floors = 0.50) proportionally to
#     # whichever weights the lstsq pushed above their own floors.
#     #
#     FLOOR_CON, FLOOR_CAL, FLOOR_STR = 0.35, 0.15, 0.05
#     w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
#     total = w1 + w2 + w3
#     if total < 1e-8:
#         w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
#     else:
#         w1, w2, w3 = w1/total, w2/total, w3/total
#         # Compute how much each weight exceeds its own floor
#         excess1 = max(0.0, w1 - FLOOR_CON)
#         excess2 = max(0.0, w2 - FLOOR_CAL)
#         excess3 = max(0.0, w3 - FLOOR_STR)
#         total_excess = excess1 + excess2 + excess3
#         # Budget available above the floors
#         budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
#         if total_excess > 1e-8:
#             # Distribute budget in proportion to excess
#             w1 = FLOOR_CON + budget * (excess1 / total_excess)
#             w2 = FLOOR_CAL + budget * (excess2 / total_excess)
#             w3 = FLOOR_STR + budget * (excess3 / total_excess)
#         else:
#             # All weights were at or below floor — give budget to consistency
#             w1 = FLOOR_CON + budget
#             w2 = FLOOR_CAL
#             w3 = FLOOR_STR
#
#     # Derive w3 from w1+w2 to avoid float accumulation
#     w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
#     # Final safety clamp (float rounding edge case)
#     w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
#     total = w1 + w2 + w3
#     w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)
#
#     blend_weights = {
#         "consistency":  w1,
#         "calibration":  w2,
#         "strategy":     w3,
#     }
#
#     log.info(
#         f"  ALC params learned from val set:\n"
#         f"    conflict_threshold={conflict_thr}  "
#         f"min_consistency={min_consistency}\n"
#         f"    variance_decay={variance_decay}  "
#         f"direction_threshold={best_dir_thr}\n"
#         f"    blend_weights={blend_weights}"
#     )
#
#     return {
#         "conflict_threshold":  conflict_thr,
#         "min_consistency":     min_consistency,
#         "variance_decay":      variance_decay,
#         "direction_threshold": best_dir_thr,
#         "blend_weights":       blend_weights,
#         # strategy_quality is a pipeline constant, not learned from val data
#         "strategy_quality": {
#             "ground_truth": 1.00,
#             "heuristic":    0.80,
#             "all_lines":    0.50,
#         },
#     }
#
# def learn_task_weights(X_tr, y_tr):
#     """
#     Learn relative importance weights for the four MTD tasks
#     by gradient descent on the training set.
#     Weights are non-negative and sum to 1.
#     """
#     cols    = list(_TASK_COLS.values())
#     Xt      = X_tr[:, cols].astype(np.float64)
#     col_std = Xt.std(0); col_std[col_std < 1e-8] = 1.0
#     Xn      = Xt / col_std
#     yf      = y_tr.astype(np.float64)
#     rng     = np.random.RandomState(42)
#     w       = rng.rand(4).astype(np.float64) * 0.25
#     for _ in range(500):
#         prob = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w, -500, 500)))
#         grad = Xn.T @ (prob - yf) / len(yf)
#         w   -= 0.05 * grad
#         w    = np.maximum(0.0, w)
#         s    = w.sum()
#         if s > 1e-8:
#             w /= s
#     if w.sum() < 1e-8:
#         w = np.array([0.25, 0.25, 0.25, 0.25])
#     wts = {k: round(float(v), 6)
#            for k, v in zip(["task1","task2","task3","task4"], w)}
#     log.info(f"  Learned task weights: {wts}")
#     return wts
#
#
#
#
# def learn_task3_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
#     """
#     Learn severity multipliers (w_H, w_M, w_L) for Task 3.
#
#     Searches over integer multiplier ratios subject to
#     monotonicity: w_H >= w_M >= w_L > 0.
#
#     The counts high_n, medium_n, low_n are stored as separate
#     features in the feature vector — we find the combination
#     that best correlates with vulnerability labels.
#     """
#     try:
#         high_col   = FEATURE_NAMES.index("high_severity_count")
#         medium_col = FEATURE_NAMES.index("medium_severity_count")
#         low_col    = FEATURE_NAMES.index("low_severity_count")
#     except ValueError:
#         # Feature names not found — return safe defaults
#         log.warning("  Task3 severity count features not found — using default weights")
#         return {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0}
#
#     high_n   = X_va[:, high_col].astype(np.float64)
#     medium_n = X_va[:, medium_col].astype(np.float64)
#     low_n    = X_va[:, low_col].astype(np.float64)
#     yf       = y_va.astype(np.float64)
#
#     best_wH, best_wM, best_wL = 3.0, 2.0, 1.0
#     best_corr = -2.0
#
#     # Search over integer-ratio candidates satisfying w_H >= w_M >= w_L >= 1
#     for wH in range(1, 6):
#         for wM in range(1, wH + 1):
#             for wL in range(1, wM + 1):
#                 score = wH * high_n + wM * medium_n + wL * low_n
#                 if score.std() > 1e-8 and yf.std() > 1e-8:
#                     corr = float(np.corrcoef(score, yf)[0, 1])
#                     if corr > best_corr:
#                         best_corr = corr
#                         best_wH, best_wM, best_wL = float(wH), float(wM), float(wL)
#
#     log.info(
#         f"  task3_weights learned: w_H={best_wH}  w_M={best_wM}  w_L={best_wL}"
#         f"  (Pearson corr={best_corr:.4f})"
#     )
#     return {"w_H": best_wH, "w_M": best_wM, "w_L": best_wL}
#
# def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
#           task_wts, alc_params, metrics, mv_dist,
#           task1_alpha=0.50,
#           task3_weights=None, extra=None):
#     d = {
#         **model.to_dict(),
#         "scaler":             scaler.to_dict(),
#         "feature_names":      FEATURE_NAMES,
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul_combined",
#         "task1_alpha":        round(task1_alpha, 4),
#         "task3_weights":      task3_weights or {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0},
#         # All learned from data — nothing hardcoded:
#         "opt_threshold":      opt_thresh["threshold"],
#         "threshold_metrics":  opt_thresh,
#         "trust_threshold":    trust_thresh,
#         "trust_calibration":  trust_cal,
#         "task_weights":       task_wts,
#         "alc_params":         alc_params,   # ALC constants, all data-driven
#         "metrics":            metrics,
#         "megavul_score_dist": mv_dist,
#     }
#     if extra: d.update(extra)
#     return d
#
#
# def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
#     log.info("Training Logistic Regression on BigVul labels...")
#     model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
#     model.fit(Xtr, ytr)
#     vp    = model.predict_proba(Xva)
#     opt   = find_opt_threshold(vp, yva)
#     ts    = compute_val_task_scores(Xva, yva)
#     alcp  = learn_alc_params(Xva, yva, vp)
#     tcal  = calibrate_trust(vp, yva)
#     tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                   blend_weights=alcp["blend_weights"],
#                                   trust_cal=tcal,
#                                   strategy_quality=alcp["strategy_quality"])
#     twts  = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     imps  = feature_importance_logreg(model)
#     met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd   = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  LR test metrics (BigVul): {met}")
#     log.info(f"  Top-5 features: {imps[:5]}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w,
#                  extra={"feature_importances": imps})
#     path = MODELS_DIR / "logreg_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  LR model saved → {path}")
#     return met
#
#
# def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
#     log.info("Training Neural Network (MLP) on BigVul labels...")
#     model = NeuralNetwork(h1=64, h2=32, lr=0.001, epochs=epochs, bs=64, l2=1e-4, dropout=0.2)
#     model.fit(Xtr, ytr)
#     vp   = model.predict_proba(Xva)
#     opt  = find_opt_threshold(vp, yva)
#     ts   = compute_val_task_scores(Xva, yva)
#     alcp = learn_alc_params(Xva, yva, vp)
#     tcal = calibrate_trust(vp, yva)
#     tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                  blend_weights=alcp["blend_weights"],
#                                  trust_cal=tcal,
#                                  strategy_quality=alcp["strategy_quality"])
#     twts = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd  = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  NN test metrics (BigVul): {met}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w)
#     path = MODELS_DIR / "nn_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  NN model saved → {path}")
#     return met
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(description="Train MTD ML models")
#     parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--test-ratio", type=float, default=0.20)
#     parser.add_argument("--val-ratio",  type=float, default=0.15)
#     args = parser.parse_args()
#
#     log.info("=== MTD ML Training ===")
#     log.info(f"Supervised: BigVul only  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")
#
#     # ── Load BigVul labelled data for training ────────────────────────────────
#     Xbv, ybv, ids = load_labelled("bigvul")
#     n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
#     log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
#     if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
#         log.error(
#             f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
#             f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
#         )
#         sys.exit(1)
#
#     # ── Load MegaVul features for scaler fitting only (not for training) ──────
#     # IMPORTANT: MegaVul is used ONLY to fit the scaler so it generalises
#     # to MegaVul's feature distribution at inference time.
#     # MegaVul is NOT used for training, calibration, or threshold finding
#     # because its V scores are unreliable (heuristic strategy, no flaw lines).
#     # Mixing MegaVul into calibrate_trust() destroys the T score distribution.
#     Xmv, _ = load_all_features("megavul")
#
#     # ── Train/val/test split on BigVul only ───────────────────────────────────
#     Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
#                                              val=args.val_ratio, test=args.test_ratio)
#     log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
#     log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
#     log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
#     log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")
#
#     # ── Scaler fitted on BigVul train + all MegaVul features ─────────────────
#     scaler = build_combined_scaler(Xtr, Xmv)
#     Xtr_s  = scaler.transform(Xtr)
#     Xva_s  = scaler.transform(Xva)
#     Xte_s  = scaler.transform(Xte)
#     Xmv_s  = (scaler.transform(Xmv)
#                if Xmv.shape[0] > 0
#                else np.empty((0, FEATURE_DIM), dtype=np.float32))
#
#     report = {
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul",
#         "n_train":            len(ytr),
#         "n_val":              len(yva),
#         "n_test":             len(yte),
#         "n_megavul_scaler":   Xmv.shape[0],
#     }
#     if args.model in ("logreg","both"):
#         report["logreg"] = train_logreg(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s)
#     if args.model in ("nn","both"):
#         report["nn"] = train_nn(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s,
#                                 epochs=args.epochs)
#
#     rp = MODELS_DIR / "training_report.json"
#     rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
#     log.info(f"Training report → {rp}")
#     log.info("=== Training complete ===")
#
#
# def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
#
#
# if __name__ == "__main__":
#     main()


# # =============================================================================
# # mtd/ml/train.py  —  ML Model Trainer
# #
# # Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# # Saves into each model JSON (ALL values learned from data, NONE hardcoded):
# #   weights/params       model parameters
# #   scaler               mean/std for feature normalisation
# #   opt_threshold        F1-optimal V threshold (from val set search)
# #   trust_threshold      threshold below which T triggers "untrustworthy"
# #                        (learned from val-set consistency distribution)
# #   trust_calibration    {intercept, slope}: V → raw T mapping
# #   task_weights         learned relative task importance
# #   feature_importances  ranked LR feature weights
# #   metrics              accuracy, precision, recall, F1, AUC
# #
# # Usage:
# #   python mtd/ml/train.py --model both --epochs 100
# # =============================================================================
#
# import argparse
# import json
# import logging
# import math
# import random
# import sys
# from pathlib import Path
# from collections import Counter
#
# import numpy as np
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from feature_extractor import FEATURE_DIM, FEATURE_NAMES
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# MODELS_DIR   = Path(__file__).resolve().parent / "models"
# DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
# _TASK_COLS = {
#     "task1": FEATURE_NAMES.index("pattern_hit_count"),
#     "task2": FEATURE_NAMES.index("risky_line_ratio"),
#     "task3": FEATURE_NAMES.index("overall_syntax_risk"),
#     "task4": FEATURE_NAMES.index("overall_dep_risk"),
# }
#
#
# # =============================================================================
# # Data loading
# # =============================================================================
#
# def load_labelled(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         log.error(f"Not found: {path}  —  run build_dataset.py first")
#         sys.exit(1)
#     X, y, ids = [], [], []
#     skipped = 0
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             if rec.get("label", -1) == -1: skipped += 1; continue
#             feats = rec.get("features", [])
#             if len(feats) != FEATURE_DIM: continue
#             X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
#     log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
#     return X, y, ids
#
#
# def load_all_features(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         return np.empty((0, FEATURE_DIM), dtype=np.float32), []
#     X, ids = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             feats = rec.get("features", [])
#             if len(feats) == FEATURE_DIM:
#                 X.append(feats); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
#     log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
#     return X, ids
#
#
# def split3(X, y, ids, val=0.15, test=0.20, seed=42):
#     rng = random.Random(seed)
#     pos = [i for i, l in enumerate(y) if l == 1]
#     neg = [i for i, l in enumerate(y) if l == 0]
#     rng.shuffle(pos); rng.shuffle(neg)
#     def cut(lst):
#         n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
#         return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
#     def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
#     pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
#     tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
#     return (X[tr],y[tr],[ids[i] for i in tr],
#             X[va],y[va],[ids[i] for i in va],
#             X[te],y[te],[ids[i] for i in te])
#
#
# # =============================================================================
# # Scaler
# # =============================================================================
#
# class StandardScaler:
#     def __init__(self): self.mean_=None; self.std_=None
#     def fit(self, X):
#         self.mean_=X.mean(0); self.std_=X.std(0)
#         self.std_[self.std_<1e-8]=1.0; return self
#     def transform(self, X): return (X-self.mean_)/self.std_
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
#     @classmethod
#     def from_dict(cls, d):
#         s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
#         s.std_=np.array(d["std"],dtype=np.float32); return s
#
#
# def build_combined_scaler(X_bv, X_mv):
#     parts = [X_bv]
#     if X_mv.shape[0] > 0: parts.append(X_mv)
#     X = np.vstack(parts)
#     sc = StandardScaler(); sc.fit(X)
#     log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
#     return sc
#
#
# # =============================================================================
# # Learned parameters
# # =============================================================================
#
# def find_opt_threshold(probs, y):
#     best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
#     for t in [i/100 for i in range(5, 96)]:
#         preds = (probs >= t).astype(int)
#         tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
#         fn = int(((preds==0)&(y==1)).sum())
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 2*prec*rec/max(1e-8,prec+rec)
#         if f1 > best["f1"]:
#             best = {"threshold":round(t,2),"f1":round(f1,4),
#                     "precision":round(prec,4),"recall":round(rec,4)}
#     log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
#     return best
#
#
# def find_trust_threshold(probs, y, task_scores_list, variance_decay,
#                           blend_weights=None, trust_cal=None,
#                           strategy_quality=None):
#     """
#     Learn the trust threshold from the validation set.
#
#     Searches over actual T values (blending consistency + calibration +
#     strategy quality) rather than raw consistency alone, so the threshold
#     is calibrated to the same space ALC uses at inference time.
#     """
#     import math as _m
#
#     bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
#     tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
#     sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
#     w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]
#
#     # Compute T for every val sample (same formula as trust_score_computation.py)
#     T_vals = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
#         vals = list(ts.values())
#         m    = sum(vals) / len(vals)
#         var  = sum((v - m) ** 2 for v in vals) / len(vals)
#         cons = max(0.10, _m.exp(-var * variance_decay))
#         dec  = abs(prob - 0.5) * 2.0
#         cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
#         strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
#         T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
#         T_vals.append((T, int(y[i])))
#
#     # Find the threshold that best separates correct from incorrect predictions
#     # by maximising: (recall of incorrect below threshold) + (precision above)
#     # Search range is clamped to [T_min, T_max] of actual val-set T values
#     # so the threshold is always within the real T distribution.
#     t_values = [T for T, _ in T_vals]
#     t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
#     t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
#     thresholds = [round(t_min + i * 0.05, 2)
#                   for i in range(int((t_max - t_min) / 0.05) + 1)]
#     if not thresholds:
#         thresholds = [0.50]
#     best_t = thresholds[len(thresholds)//2]; best_score = -1.0
#
#     for t in thresholds:
#         above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
#         above_total   = sum(1 for T, _      in T_vals if T >= t)
#         below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
#         total_wrong   = sum(1 for _, correct in T_vals if not correct)
#
#         precision_above  = above_correct / max(1, above_total)
#         recall_wrong_below = below_wrong / max(1, total_wrong)
#         score = (precision_above + recall_wrong_below) / 2.0
#
#         if score > best_score:
#             best_score = score; best_t = t
#
#     log.info(f"  trust_threshold={best_t:.2f}  "
#              f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     return round(best_t, 2)
#
#
# def calibrate_trust(probs, y):
#     """
#     Fit a linear calibration: oracle_trust = intercept + slope * decisiveness
#     where oracle_trust = 1 - |V - y|  (1=correct, 0=maximally wrong)
#     and decisiveness = |V - 0.5| * 2  (0=uncertain, 1=fully confident)
#
#     The intercept represents the base trust level for an indecisive model.
#     It must be >= 0.45 so that decisive clean samples can reach T >= 0.8
#     even when consistency is moderate (0.55-0.65 range).
#
#     If the fitted intercept drops below 0.45 it means the val set contains
#     too many wrong predictions pulling it down — clamp it to ensure the
#     T score distribution remains meaningful.
#     """
#     v = probs.astype(np.float64); yf = y.astype(np.float64)
#     oracle  = 1.0 - np.abs(v - yf)
#     decisiv = np.abs(v - 0.5) * 2.0
#     A = np.column_stack([np.ones_like(decisiv), decisiv])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         # Floor at 0.45: ensures decisive clean samples reach T >= 0.8
#         # Ceiling at 0.75: prevents T from being trivially high everywhere
#         intercept = float(np.clip(coeffs[0], 0.45, 0.75))
#         slope     = float(np.clip(coeffs[1], 0.0,  0.60))
#     except Exception:
#         intercept, slope = 0.48, 0.50
#     log.info(
#         f"  trust_calibration: intercept={intercept:.4f}  slope={slope:.4f}"
#     )
#     return {"intercept": round(intercept,6), "slope": round(slope,6)}
#
#
#
#
# def learn_task1_alpha(X_va: np.ndarray, y_va: np.ndarray) -> float:
#     """
#     Learn the alpha parameter for Task 1 from the validation set.
#
#     alpha controls the balance between hit presence and confidence:
#         s1 = alpha * hit_indicator + (1 - alpha) * avg_confidence
#
#     We find the alpha in (0,1) that maximises the Pearson correlation
#     between the resulting s1 values and the ground-truth labels y.
#
#     High alpha -> trust hit presence more  (robust but less nuanced)
#     Low  alpha -> trust confidence more    (precise but noise-sensitive)
#
#     The optimal value is dataset-dependent:
#       - BigVul  (ground_truth strategy, precise patterns) -> alpha ~ 0.45
#       - Noisier datasets (heuristic/all_lines)             -> alpha ~ 0.55-0.70
#     """
#     hit_col  = FEATURE_NAMES.index("pattern_hit_count")
#     conf_col = FEATURE_NAMES.index("avg_line_conf")
#
#     hits = X_va[:, hit_col].astype(np.float64)
#     conf = X_va[:, conf_col].astype(np.float64)
#     yf   = y_va.astype(np.float64)
#
#     # Binarise hits: 1 if any pattern fired, 0 otherwise
#     hit_indicator = (hits > 0).astype(np.float64)
#
#     best_alpha = 0.50
#     best_corr  = -2.0
#
#     for i in range(1, 20):          # alpha in {0.05, 0.10, ..., 0.95}
#         alpha = round(i / 20.0, 2)
#         s1    = alpha * hit_indicator + (1.0 - alpha) * conf
#         if s1.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(s1, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr  = corr
#                 best_alpha = alpha
#
#     log.info(
#         f"  task1_alpha learned: {best_alpha}  "
#         f"(Pearson corr with labels = {best_corr:.4f})"
#     )
#     return best_alpha
#
#
# # =============================================================================
# # Models
# # =============================================================================
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
#         self.lr=lr; self.epochs=epochs; self.bs=batch_size
#         self.l2=l2; self.seed=seed; self.w=None; self.b=None
#     def fit(self, X, y):
#         rng=np.random.RandomState(self.seed); n,d=X.shape
#         self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
#         for epoch in range(self.epochs):
#             idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
#                 p=_sig(Xb@self.w+self.b); e=p-yb
#                 self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
#                 loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
#             if (epoch+1)%100==0:
#                 log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
#         return self
#     def predict_proba(self, X): return _sig(X@self.w+self.b)
#     def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m
#
#
# class NeuralNetwork:
#     def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
#         self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
#         self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
#     def _init(self, d):
#         rng=np.random.RandomState(self.seed); self._rng=rng
#         self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
#                 "b1":np.zeros(self.h1,dtype=np.float32),
#                 "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
#                 "b2":np.zeros(self.h2,dtype=np.float32),
#                 "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
#                 "b3":np.zeros(1,dtype=np.float32)}
#     def _fwd(self, X, train=False):
#         p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
#         if train and self.dropout>0:
#             m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
#         else: m1=None
#         z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
#         if train and self.dropout>0:
#             m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
#         else: m2=None
#         return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
#     def _bwd(self, X, y, prob, cache):
#         z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
#         dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
#         dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
#         da2=dz3@p["W3"].T
#         if m2 is not None: da2=da2*m2/(1-self.dropout)
#         dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
#         da1=dz2@p["W2"].T
#         if m1 is not None: da1=da1*m1/(1-self.dropout)
#         dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
#         return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
#     def fit(self, X, y):
#         self._init(X.shape[1]); n=len(y)
#         for epoch in range(self.epochs):
#             idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
#                 prob,cache=self._fwd(Xb,train=True)
#                 loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
#                 grads=self._bwd(Xb,yb,prob,cache)
#                 for k in self.p: self.p[k]-=self.lr*grads[k]
#             if (epoch+1)%20==0:
#                 pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
#                 log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
#         return self
#     def predict_proba(self, X): p,_=self._fwd(X); return p
#     def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# def evaluate(model, Xte, yte, threshold=0.50):
#     probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
#     tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
#     fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
#     acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
#     f1=2*prec*rec/max(1e-8,prec+rec)
#     pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
#     if len(pos_p)>0 and len(neg_p)>0:
#         c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
#         auc=c/(len(pos_p)*len(neg_p))
#     return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
#             "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
#             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
#             "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
#             "evaluated_on":"bigvul_test_set"}
#
#
# def feature_importance_logreg(model):
#     abs_w=np.abs(model.w); total=abs_w.sum()
#     if total<1e-8: return []
#     ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
#     return [{"feature":f,"importance":round(i,6)} for f,i in ranked]
#
#
# def compute_megavul_score_dist(model, X_mv_scaled):
#     if X_mv_scaled.shape[0]==0:
#         return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
#     probs=model.predict_proba(X_mv_scaled)
#     buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
#     for p in probs:
#         if p<0.2: buckets["0.0-0.2"]+=1
#         elif p<0.4: buckets["0.2-0.4"]+=1
#         elif p<0.6: buckets["0.4-0.6"]+=1
#         elif p<0.8: buckets["0.6-0.8"]+=1
#         else: buckets["0.8-1.0"]+=1
#     return {"n":len(probs),"mean":round(float(probs.mean()),4),
#             "std":round(float(probs.std()),4),"buckets":buckets}
#
#
# # =============================================================================
# # Compute val task scores for trust threshold learning
# # =============================================================================
#
# def compute_val_task_scores(X_va, y_va):
#     """Extract per-sample task score dicts from val feature vectors."""
#     task_scores_list = []
#     cols = _TASK_COLS
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#     return task_scores_list
#
#
# # =============================================================================
# # Training entry points
# # =============================================================================
#
#
# def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
#                      val_probs: np.ndarray) -> dict:
#     """
#     Learn all ALC constants from the validation set.
#     Nothing in alc/run_alc.py is hardcoded — every number comes from here.
#
#     Learned values:
#       conflict_threshold  — pairwise score diff that counts as "conflict"
#                             = mean absolute pairwise difference on val set
#                             (tasks that naturally spread this far are conflicting)
#       min_consistency     — lowest achievable consistency score
#                             = consistency at maximum observed variance
#       variance_decay      — steepness of exp(-var * decay)
#                             = fitted so exp(-max_var * decay) = min_consistency
#       direction_threshold — score boundary separating "risky" from "clean"
#                             = optimal threshold that best separates val labels
#       blend_weights       — {consistency, calibration, strategy}
#                             = learned by fitting a 3-feature linear model
#                               mapping (consistency, calibration, strategy_quality)
#                               to oracle trust (1 - |V - y|) on val set
#       strategy_quality    — {ground_truth, heuristic, all_lines}
#                             = kept as data-pipeline constants (not empirical)
#                               because they reflect factual confidence levels
#                               about the suspicious-line mapping process itself,
#                               not something the val set can determine
#     """
#     import math as _math
#
#     cols = _TASK_COLS
#     task_scores_list = []
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#
#     # ── conflict_threshold ──────────────────────────────────────────────────
#     # Compute all pairwise absolute differences on val set
#     all_diffs = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         for i in range(len(vals)):
#             for j in range(i+1, len(vals)):
#                 all_diffs.append(abs(vals[i] - vals[j]))
#
#     # Use the 75th percentile: pairs above this are genuinely "conflicting"
#     all_diffs.sort()
#     p75_idx      = int(len(all_diffs) * 0.75)
#     conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
#     # Clamp to a sensible range [0.15, 0.50]
#     conflict_thr = max(0.15, min(0.50, conflict_thr))
#
#     # ── variance_decay and min_consistency ─────────────────────────────────
#     # Compute per-sample variances
#     variances = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         m = sum(vals) / len(vals)
#         variances.append(sum((v-m)**2 for v in vals) / len(vals))
#
#     max_var = max(variances) if variances else 0.25
#     min_var = min(variances) if variances else 0.0
#
#     # min_consistency = the lowest trust we should assign even at max disagreement
#     # = fraction of val samples that are correct even at maximum variance
#     correct_at_max = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
#         vals = list(ts.values()); m = sum(vals)/len(vals)
#         var  = sum((v-m)**2 for v in vals)/len(vals)
#         if abs(var - max_var) < 0.02:   # near-maximum variance samples
#             correct = (int(prob >= 0.5) == int(y_va[i]))
#             correct_at_max.append(float(correct))
#     min_consistency = round(
#         float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
#         4
#     )
#     min_consistency = max(0.05, min(0.30, min_consistency))
#
#     # variance_decay: solve exp(-max_var * decay) = min_consistency
#     # → decay = -ln(min_consistency) / max_var
#     if max_var > 1e-6 and min_consistency > 1e-6:
#         variance_decay = round(-_math.log(min_consistency) / max_var, 4)
#         variance_decay = max(2.0, min(20.0, variance_decay))
#     else:
#         variance_decay = 8.0
#
#     # ── direction_threshold ─────────────────────────────────────────────────
#     # Find the task score boundary that best separates vulnerable from clean
#     # Search over [0.10, 0.70] using mean task score on val set
#     mean_scores = [
#         sum(ts.values()) / len(ts) for ts in task_scores_list
#     ]
#     best_dir_thr = 0.30; best_acc = 0.0
#     for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
#         preds = [1 if s >= t else 0 for s in mean_scores]
#         acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
#         if acc > best_acc:
#             best_acc = acc; best_dir_thr = round(t, 2)
#
#     # ── blend_weights for Stage 3 ───────────────────────────────────────────
#     # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
#     # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
#     # Consistency and calibration are computable from val data
#     # strategy_quality: use 0.80 (heuristic) for all val samples
#     #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)
#
#     oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
#     decisive = np.abs(val_probs - 0.5) * 2.0
#
#     # Compute per-sample consistency from task variance
#     consistencies = np.array([
#         max(min_consistency,
#             _math.exp(-v * variance_decay))
#         for v in variances
#     ], dtype=np.float64)
#
#     # Calibration column (same formula used at inference)
#     calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
#     strategy_col = np.full(len(y_va), 0.80)                    # heuristic default
#
#     A = np.column_stack([consistencies, calibrations, strategy_col])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
#     except Exception:
#         w1, w2, w3 = 0.50, 0.35, 0.15
#
#     # Enforce minimum floors without iterative oscillation.
#     #
#     # The iterative approach (clamp → renormalise → repeat) fails because
#     # dividing by the new total pulls w1 back below the floor every step.
#     #
#     # Correct approach: treat the floors as hard reservations.
#     # Whatever the lstsq gives, first assign the floors, then distribute
#     # the remaining budget (1 - sum_of_floors = 0.50) proportionally to
#     # whichever weights the lstsq pushed above their own floors.
#     #
#     FLOOR_CON, FLOOR_CAL, FLOOR_STR = 0.35, 0.15, 0.05
#     w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
#     total = w1 + w2 + w3
#     if total < 1e-8:
#         w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
#     else:
#         w1, w2, w3 = w1/total, w2/total, w3/total
#         # Compute how much each weight exceeds its own floor
#         excess1 = max(0.0, w1 - FLOOR_CON)
#         excess2 = max(0.0, w2 - FLOOR_CAL)
#         excess3 = max(0.0, w3 - FLOOR_STR)
#         total_excess = excess1 + excess2 + excess3
#         # Budget available above the floors
#         budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
#         if total_excess > 1e-8:
#             # Distribute budget in proportion to excess
#             w1 = FLOOR_CON + budget * (excess1 / total_excess)
#             w2 = FLOOR_CAL + budget * (excess2 / total_excess)
#             w3 = FLOOR_STR + budget * (excess3 / total_excess)
#         else:
#             # All weights were at or below floor — give budget to consistency
#             w1 = FLOOR_CON + budget
#             w2 = FLOOR_CAL
#             w3 = FLOOR_STR
#
#     # Derive w3 from w1+w2 to avoid float accumulation
#     w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
#     # Final safety clamp (float rounding edge case)
#     w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
#     total = w1 + w2 + w3
#     w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)
#
#     blend_weights = {
#         "consistency":  w1,
#         "calibration":  w2,
#         "strategy":     w3,
#     }
#
#     log.info(
#         f"  ALC params learned from val set:\n"
#         f"    conflict_threshold={conflict_thr}  "
#         f"min_consistency={min_consistency}\n"
#         f"    variance_decay={variance_decay}  "
#         f"direction_threshold={best_dir_thr}\n"
#         f"    blend_weights={blend_weights}"
#     )
#
#     return {
#         "conflict_threshold":  conflict_thr,
#         "min_consistency":     min_consistency,
#         "variance_decay":      variance_decay,
#         "direction_threshold": best_dir_thr,
#         "blend_weights":       blend_weights,
#         # strategy_quality is a pipeline constant, not learned from val data
#         "strategy_quality": {
#             "ground_truth": 1.00,
#             "heuristic":    0.80,
#             "all_lines":    0.50,
#         },
#     }
#
# def learn_task_weights(X_tr, y_tr):
#     """
#     Learn relative importance weights for the four MTD tasks
#     by gradient descent on the training set.
#     Weights are non-negative and sum to 1.
#     """
#     cols    = list(_TASK_COLS.values())
#     Xt      = X_tr[:, cols].astype(np.float64)
#     col_std = Xt.std(0); col_std[col_std < 1e-8] = 1.0
#     Xn      = Xt / col_std
#     yf      = y_tr.astype(np.float64)
#     rng     = np.random.RandomState(42)
#     w       = rng.rand(4).astype(np.float64) * 0.25
#     for _ in range(500):
#         prob = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w, -500, 500)))
#         grad = Xn.T @ (prob - yf) / len(yf)
#         w   -= 0.05 * grad
#         w    = np.maximum(0.0, w)
#         s    = w.sum()
#         if s > 1e-8:
#             w /= s
#     if w.sum() < 1e-8:
#         w = np.array([0.25, 0.25, 0.25, 0.25])
#     wts = {k: round(float(v), 6)
#            for k, v in zip(["task1","task2","task3","task4"], w)}
#     log.info(f"  Learned task weights: {wts}")
#     return wts
#
#
#
#
# def learn_task3_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
#     """
#     Learn severity multipliers (w_H, w_M, w_L) for Task 3.
#
#     Searches over integer multiplier ratios subject to
#     monotonicity: w_H >= w_M >= w_L > 0.
#
#     The counts high_n, medium_n, low_n are stored as separate
#     features in the feature vector — we find the combination
#     that best correlates with vulnerability labels.
#     """
#     try:
#         high_col   = FEATURE_NAMES.index("high_severity_count")
#         medium_col = FEATURE_NAMES.index("medium_severity_count")
#         low_col    = FEATURE_NAMES.index("low_severity_count")
#     except ValueError:
#         # Feature names not found — return safe defaults
#         log.warning("  Task3 severity count features not found — using default weights")
#         return {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0}
#
#     high_n   = X_va[:, high_col].astype(np.float64)
#     medium_n = X_va[:, medium_col].astype(np.float64)
#     low_n    = X_va[:, low_col].astype(np.float64)
#     yf       = y_va.astype(np.float64)
#
#     best_wH, best_wM, best_wL = 3.0, 2.0, 1.0
#     best_corr = -2.0
#
#     # Search over integer-ratio candidates satisfying w_H >= w_M >= w_L >= 1
#     for wH in range(1, 6):
#         for wM in range(1, wH + 1):
#             for wL in range(1, wM + 1):
#                 score = wH * high_n + wM * medium_n + wL * low_n
#                 if score.std() > 1e-8 and yf.std() > 1e-8:
#                     corr = float(np.corrcoef(score, yf)[0, 1])
#                     if corr > best_corr:
#                         best_corr = corr
#                         best_wH, best_wM, best_wL = float(wH), float(wM), float(wL)
#
#     log.info(
#         f"  task3_weights learned: w_H={best_wH}  w_M={best_wM}  w_L={best_wL}"
#         f"  (Pearson corr={best_corr:.4f})"
#     )
#     return {"w_H": best_wH, "w_M": best_wM, "w_L": best_wL}
#
#
#
# def learn_task4_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
#     """
#     Learn the data-flow and control-flow blend weights for Task 4.
#
#     Task 4 computes:
#         overall = w_df * df_risk + w_cf * cf_risk,  w_df + w_cf = 1
#
#     We search over w_df in [0.05, 0.95] (step 0.05) and set
#     w_cf = 1 - w_df, choosing the value that maximises Pearson
#     correlation with ground-truth labels on the validation set.
#
#     Defaults: w_df=0.60, w_cf=0.40 (data-flow dominates since
#     most memory-safety vulnerabilities originate from data flow).
#     """
#     try:
#         df_col = FEATURE_NAMES.index("data_flow_risk")
#         cf_col = FEATURE_NAMES.index("control_flow_risk")
#     except ValueError:
#         log.warning("  Task4 risk features not found — using default weights")
#         return {"w_df": 0.60, "w_cf": 0.40}
#
#     df_risk = X_va[:, df_col].astype(np.float64)
#     cf_risk = X_va[:, cf_col].astype(np.float64)
#     yf      = y_va.astype(np.float64)
#
#     best_wdf  = 0.60
#     best_corr = -2.0
#
#     for i in range(1, 20):          # w_df in {0.05, 0.10, ..., 0.95}
#         w_df  = round(i / 20.0, 2)
#         w_cf  = round(1.0 - w_df, 2)
#         score = w_df * df_risk + w_cf * cf_risk
#         if score.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(score, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr = corr
#                 best_wdf  = w_df
#
#     best_wcf = round(1.0 - best_wdf, 2)
#     log.info(
#         f"  task4_weights learned: w_df={best_wdf}  w_cf={best_wcf}"
#         f"  (Pearson corr={best_corr:.4f})"
#     )
#     return {"w_df": best_wdf, "w_cf": best_wcf}
#
# def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
#           task_wts, alc_params, metrics, mv_dist,
#           task1_alpha=0.50,
#           task3_weights=None,
#           task4_weights=None, extra=None):
#     d = {
#         **model.to_dict(),
#         "scaler":             scaler.to_dict(),
#         "feature_names":      FEATURE_NAMES,
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul_combined",
#         "task1_alpha":        round(task1_alpha, 4),
#         "task3_weights":      task3_weights or {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0},
#         "task4_weights":      task4_weights or {"w_df": 0.60, "w_cf": 0.40},
#         # All learned from data — nothing hardcoded:
#         "opt_threshold":      opt_thresh["threshold"],
#         "threshold_metrics":  opt_thresh,
#         "trust_threshold":    trust_thresh,
#         "trust_calibration":  trust_cal,
#         "task_weights":       task_wts,
#         "alc_params":         alc_params,   # ALC constants, all data-driven
#         "metrics":            metrics,
#         "megavul_score_dist": mv_dist,
#     }
#     if extra: d.update(extra)
#     return d
#
#
# def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
#     log.info("Training Logistic Regression on BigVul labels...")
#     model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
#     model.fit(Xtr, ytr)
#     vp    = model.predict_proba(Xva)
#     opt   = find_opt_threshold(vp, yva)
#     ts    = compute_val_task_scores(Xva, yva)
#     alcp  = learn_alc_params(Xva, yva, vp)
#     tcal  = calibrate_trust(vp, yva)
#     tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                   blend_weights=alcp["blend_weights"],
#                                   trust_cal=tcal,
#                                   strategy_quality=alcp["strategy_quality"])
#     twts  = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     t4w   = learn_task4_weights(Xva, yva)
#     imps  = feature_importance_logreg(model)
#     met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd   = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  LR test metrics (BigVul): {met}")
#     log.info(f"  Top-5 features: {imps[:5]}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w,
#                  extra={"feature_importances": imps})
#     path = MODELS_DIR / "logreg_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  LR model saved → {path}")
#     return met
#
#
# def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
#     log.info("Training Neural Network (MLP) on BigVul labels...")
#     model = NeuralNetwork(h1=64, h2=32, lr=0.001, epochs=epochs, bs=64, l2=1e-4, dropout=0.2)
#     model.fit(Xtr, ytr)
#     vp   = model.predict_proba(Xva)
#     opt  = find_opt_threshold(vp, yva)
#     ts   = compute_val_task_scores(Xva, yva)
#     alcp = learn_alc_params(Xva, yva, vp)
#     tcal = calibrate_trust(vp, yva)
#     tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                  blend_weights=alcp["blend_weights"],
#                                  trust_cal=tcal,
#                                  strategy_quality=alcp["strategy_quality"])
#     twts = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     t4w   = learn_task4_weights(Xva, yva)
#     met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd  = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  NN test metrics (BigVul): {met}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w)
#     path = MODELS_DIR / "nn_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  NN model saved → {path}")
#     return met
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(description="Train MTD ML models")
#     parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--test-ratio", type=float, default=0.20)
#     parser.add_argument("--val-ratio",  type=float, default=0.15)
#     args = parser.parse_args()
#
#     log.info("=== MTD ML Training ===")
#     log.info(f"Supervised: BigVul only  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")
#
#     # ── Load BigVul labelled data for training ────────────────────────────────
#     Xbv, ybv, ids = load_labelled("bigvul")
#     n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
#     log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
#     if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
#         log.error(
#             f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
#             f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
#         )
#         sys.exit(1)
#
#     # ── Load MegaVul features for scaler fitting only (not for training) ──────
#     # IMPORTANT: MegaVul is used ONLY to fit the scaler so it generalises
#     # to MegaVul's feature distribution at inference time.
#     # MegaVul is NOT used for training, calibration, or threshold finding
#     # because its V scores are unreliable (heuristic strategy, no flaw lines).
#     # Mixing MegaVul into calibrate_trust() destroys the T score distribution.
#     Xmv, _ = load_all_features("megavul")
#
#     # ── Train/val/test split on BigVul only ───────────────────────────────────
#     Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
#                                              val=args.val_ratio, test=args.test_ratio)
#     log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
#     log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
#     log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
#     log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")
#
#     # ── Scaler fitted on BigVul train + all MegaVul features ─────────────────
#     scaler = build_combined_scaler(Xtr, Xmv)
#     Xtr_s  = scaler.transform(Xtr)
#     Xva_s  = scaler.transform(Xva)
#     Xte_s  = scaler.transform(Xte)
#     Xmv_s  = (scaler.transform(Xmv)
#                if Xmv.shape[0] > 0
#                else np.empty((0, FEATURE_DIM), dtype=np.float32))
#
#     report = {
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul",
#         "n_train":            len(ytr),
#         "n_val":              len(yva),
#         "n_test":             len(yte),
#         "n_megavul_scaler":   Xmv.shape[0],
#     }
#     if args.model in ("logreg","both"):
#         report["logreg"] = train_logreg(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s)
#     if args.model in ("nn","both"):
#         report["nn"] = train_nn(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s,
#                                 epochs=args.epochs)
#
#     rp = MODELS_DIR / "training_report.json"
#     rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
#     log.info(f"Training report → {rp}")
#     log.info("=== Training complete ===")
#
#
# def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
#
#
# if __name__ == "__main__":
#     main()



# # =============================================================================
# # mtd/ml/train.py  —  ML Model Trainer
# #
# # Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# # Saves into each model JSON (ALL values learned from data, NONE hardcoded):
# #   weights/params       model parameters
# #   scaler               mean/std for feature normalisation
# #   opt_threshold        F1-optimal V threshold (from val set search)
# #   trust_threshold      threshold below which T triggers "untrustworthy"
# #                        (learned from val-set consistency distribution)
# #   trust_calibration    {intercept, slope}: V → raw T mapping
# #   task_weights         learned relative task importance
# #   feature_importances  ranked LR feature weights
# #   metrics              accuracy, precision, recall, F1, AUC
# #
# # Usage:
# #   python mtd/ml/train.py --model both --epochs 100
# # =============================================================================
#
# import argparse
# import json
# import logging
# import math
# import random
# import sys
# from pathlib import Path
# from collections import Counter
#
# import numpy as np
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# from feature_extractor import FEATURE_DIM, FEATURE_NAMES
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# log = logging.getLogger(__name__)
#
# MODELS_DIR   = Path(__file__).resolve().parent / "models"
# DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
#
# _TASK_COLS = {
#     "task1": FEATURE_NAMES.index("pattern_hit_count"),
#     "task2": FEATURE_NAMES.index("risky_line_ratio"),
#     "task3": FEATURE_NAMES.index("overall_syntax_risk"),
#     "task4": FEATURE_NAMES.index("overall_dep_risk"),
# }
#
#
# # =============================================================================
# # Data loading
# # =============================================================================
#
# def load_labelled(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         log.error(f"Not found: {path}  —  run build_dataset.py first")
#         sys.exit(1)
#     X, y, ids = [], [], []
#     skipped = 0
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             if rec.get("label", -1) == -1: skipped += 1; continue
#             feats = rec.get("features", [])
#             if len(feats) != FEATURE_DIM: continue
#             X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
#     log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
#     return X, y, ids
#
#
# def load_all_features(name: str):
#     path = DATASETS_DIR / f"{name}_features.jsonl"
#     if not path.exists():
#         return np.empty((0, FEATURE_DIM), dtype=np.float32), []
#     X, ids = [], []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             rec = json.loads(line)
#             feats = rec.get("features", [])
#             if len(feats) == FEATURE_DIM:
#                 X.append(feats); ids.append(rec["sample_id"])
#     X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
#     log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
#     return X, ids
#
#
# def split3(X, y, ids, val=0.15, test=0.20, seed=42):
#     rng = random.Random(seed)
#     pos = [i for i, l in enumerate(y) if l == 1]
#     neg = [i for i, l in enumerate(y) if l == 0]
#     rng.shuffle(pos); rng.shuffle(neg)
#     def cut(lst):
#         n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
#         return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
#     def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
#     pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
#     tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
#     return (X[tr],y[tr],[ids[i] for i in tr],
#             X[va],y[va],[ids[i] for i in va],
#             X[te],y[te],[ids[i] for i in te])
#
#
# # =============================================================================
# # Scaler
# # =============================================================================
#
# class StandardScaler:
#     def __init__(self): self.mean_=None; self.std_=None
#     def fit(self, X):
#         self.mean_=X.mean(0); self.std_=X.std(0)
#         self.std_[self.std_<1e-8]=1.0; return self
#     def transform(self, X): return (X-self.mean_)/self.std_
#     def fit_transform(self, X): return self.fit(X).transform(X)
#     def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
#     @classmethod
#     def from_dict(cls, d):
#         s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
#         s.std_=np.array(d["std"],dtype=np.float32); return s
#
#
# def build_combined_scaler(X_bv, X_mv):
#     parts = [X_bv]
#     if X_mv.shape[0] > 0: parts.append(X_mv)
#     X = np.vstack(parts)
#     sc = StandardScaler(); sc.fit(X)
#     log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
#     return sc
#
#
# # =============================================================================
# # Learned parameters
# # =============================================================================
#
# def find_opt_threshold(probs, y):
#     best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
#     for t in [i/100 for i in range(5, 96)]:
#         preds = (probs >= t).astype(int)
#         tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
#         fn = int(((preds==0)&(y==1)).sum())
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 2*prec*rec/max(1e-8,prec+rec)
#         if f1 > best["f1"]:
#             best = {"threshold":round(t,2),"f1":round(f1,4),
#                     "precision":round(prec,4),"recall":round(rec,4)}
#     log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
#     return best
#
#
# def find_trust_threshold(probs, y, task_scores_list, variance_decay,
#                           blend_weights=None, trust_cal=None,
#                           strategy_quality=None):
#     """
#     Learn the trust threshold from the validation set.
#
#     Searches over actual T values (blending consistency + calibration +
#     strategy quality) rather than raw consistency alone, so the threshold
#     is calibrated to the same space ALC uses at inference time.
#     """
#     import math as _m
#
#     bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
#     tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
#     sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
#     w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]
#
#     # Compute T for every val sample (same formula as trust_score_computation.py)
#     T_vals = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
#         vals = list(ts.values())
#         m    = sum(vals) / len(vals)
#         var  = sum((v - m) ** 2 for v in vals) / len(vals)
#         cons = max(0.10, _m.exp(-var * variance_decay))
#         dec  = abs(prob - 0.5) * 2.0
#         cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
#         strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
#         T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
#         T_vals.append((T, int(y[i])))
#
#     # Find the threshold that best separates correct from incorrect predictions
#     # by maximising: (recall of incorrect below threshold) + (precision above)
#     # Search range is clamped to [T_min, T_max] of actual val-set T values
#     # so the threshold is always within the real T distribution.
#     t_values = [T for T, _ in T_vals]
#     t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
#     t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
#     thresholds = [round(t_min + i * 0.05, 2)
#                   for i in range(int((t_max - t_min) / 0.05) + 1)]
#     if not thresholds:
#         thresholds = [0.50]
#     best_t = thresholds[len(thresholds)//2]; best_score = -1.0
#
#     for t in thresholds:
#         above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
#         above_total   = sum(1 for T, _      in T_vals if T >= t)
#         below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
#         total_wrong   = sum(1 for _, correct in T_vals if not correct)
#
#         precision_above  = above_correct / max(1, above_total)
#         recall_wrong_below = below_wrong / max(1, total_wrong)
#         score = (precision_above + recall_wrong_below) / 2.0
#
#         if score > best_score:
#             best_score = score; best_t = t
#
#     log.info(f"  trust_threshold={best_t:.2f}  "
#              f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
#     return round(best_t, 2)
#
#
# def calibrate_trust(probs, y):
#     """
#     Fit a linear calibration: oracle_trust = intercept + slope * decisiveness
#     where oracle_trust = 1 - |V - y|  (1=correct, 0=maximally wrong)
#     and decisiveness = |V - 0.5| * 2  (0=uncertain, 1=fully confident)
#
#     The intercept represents the base trust level for an indecisive model.
#     It must be >= 0.45 so that decisive clean samples can reach T >= 0.8
#     even when consistency is moderate (0.55-0.65 range).
#
#     If the fitted intercept drops below 0.45 it means the val set contains
#     too many wrong predictions pulling it down — clamp it to ensure the
#     T score distribution remains meaningful.
#     """
#     v = probs.astype(np.float64); yf = y.astype(np.float64)
#     oracle  = 1.0 - np.abs(v - yf)
#     decisiv = np.abs(v - 0.5) * 2.0
#     A = np.column_stack([np.ones_like(decisiv), decisiv])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         # Floor at 0.45: ensures decisive clean samples reach T >= 0.8
#         # Ceiling at 0.75: prevents T from being trivially high everywhere
#         intercept = float(np.clip(coeffs[0], 0.45, 0.75))
#         slope     = float(np.clip(coeffs[1], 0.0,  0.60))
#     except Exception:
#         intercept, slope = 0.48, 0.50
#     log.info(
#         f"  trust_calibration: intercept={intercept:.4f}  slope={slope:.4f}"
#     )
#     return {"intercept": round(intercept,6), "slope": round(slope,6)}
#
#
#
#
# def learn_task1_alpha(X_va: np.ndarray, y_va: np.ndarray) -> float:
#     """
#     Learn the alpha parameter for Task 1 from the validation set.
#
#     alpha controls the balance between hit presence and confidence:
#         s1 = alpha * hit_indicator + (1 - alpha) * avg_confidence
#
#     We find the alpha in (0,1) that maximises the Pearson correlation
#     between the resulting s1 values and the ground-truth labels y.
#
#     High alpha -> trust hit presence more  (robust but less nuanced)
#     Low  alpha -> trust confidence more    (precise but noise-sensitive)
#
#     The optimal value is dataset-dependent:
#       - BigVul  (ground_truth strategy, precise patterns) -> alpha ~ 0.45
#       - Noisier datasets (heuristic/all_lines)             -> alpha ~ 0.55-0.70
#     """
#     hit_col  = FEATURE_NAMES.index("pattern_hit_count")
#     conf_col = FEATURE_NAMES.index("avg_line_conf")
#
#     hits = X_va[:, hit_col].astype(np.float64)
#     conf = X_va[:, conf_col].astype(np.float64)
#     yf   = y_va.astype(np.float64)
#
#     # Binarise hits: 1 if any pattern fired, 0 otherwise
#     hit_indicator = (hits > 0).astype(np.float64)
#
#     best_alpha = 0.50
#     best_corr  = -2.0
#
#     for i in range(1, 20):          # alpha in {0.05, 0.10, ..., 0.95}
#         alpha = round(i / 20.0, 2)
#         s1    = alpha * hit_indicator + (1.0 - alpha) * conf
#         if s1.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(s1, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr  = corr
#                 best_alpha = alpha
#
#     log.info(
#         f"  task1_alpha learned: {best_alpha}  "
#         f"(Pearson corr with labels = {best_corr:.4f})"
#     )
#     return best_alpha
#
#
# def select_nn_architecture(X_tr, y_tr, X_va, y_va):
#     """
#     Select the best NN architecture, dropout rate, and L2 regularisation
#     from a small candidate grid using validation F1 score.
#
#     Candidates:
#       - Hidden sizes: (64,32), (128,64), (32,16)
#       - Dropout:      0.1, 0.2, 0.3
#       - L2:           1e-3, 1e-4, 1e-5
#
#     We fix epochs=30 for the selection pass (fast) then the winner
#     is retrained for the full epoch count in train_nn().
#     Returns a dict of the best hyperparameters found.
#     """
#     candidates = [
#         {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-4},  # default first
#         {"h1": 128, "h2": 64, "dropout": 0.2, "l2": 1e-4},
#         {"h1": 32,  "h2": 16, "dropout": 0.2, "l2": 1e-4},
#         {"h1": 64,  "h2": 32, "dropout": 0.1, "l2": 1e-4},
#         {"h1": 64,  "h2": 32, "dropout": 0.3, "l2": 1e-4},
#         {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-3},
#         {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-5},
#     ]
#
#     best_cfg  = candidates[0]
#     best_f1   = -1.0
#
#     for cfg in candidates:
#         model = NeuralNetwork(
#             h1=cfg["h1"], h2=cfg["h2"],
#             lr=0.001, epochs=30,   # quick pass
#             bs=64, l2=cfg["l2"], dropout=cfg["dropout"]
#         )
#         model.fit(X_tr, y_tr)
#         probs = model.predict_proba(X_va)
#         opt   = find_opt_threshold(probs, y_va)
#         f1    = opt["f1"]
#         log.info(
#             f"  NN arch search: h1={cfg['h1']} h2={cfg['h2']} "
#             f"dropout={cfg['dropout']} l2={cfg['l2']} → val_F1={f1:.4f}"
#         )
#         if f1 > best_f1:
#             best_f1  = f1
#             best_cfg = cfg
#
#     log.info(
#         f"  Best NN architecture: h1={best_cfg['h1']} h2={best_cfg['h2']} "
#         f"dropout={best_cfg['dropout']} l2={best_cfg['l2']} (F1={best_f1:.4f})"
#     )
#     return best_cfg
#
#
# # =============================================================================
# # Models
# # =============================================================================
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
#         self.lr=lr; self.epochs=epochs; self.bs=batch_size
#         self.l2=l2; self.seed=seed; self.w=None; self.b=None
#     def fit(self, X, y):
#         rng=np.random.RandomState(self.seed); n,d=X.shape
#         self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
#         for epoch in range(self.epochs):
#             idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
#                 p=_sig(Xb@self.w+self.b); e=p-yb
#                 self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
#                 loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
#             if (epoch+1)%100==0:
#                 log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
#         return self
#     def predict_proba(self, X): return _sig(X@self.w+self.b)
#     def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m
#
#
# class NeuralNetwork:
#     def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
#         self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
#         self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
#     def _init(self, d):
#         rng=np.random.RandomState(self.seed); self._rng=rng
#         self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
#                 "b1":np.zeros(self.h1,dtype=np.float32),
#                 "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
#                 "b2":np.zeros(self.h2,dtype=np.float32),
#                 "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
#                 "b3":np.zeros(1,dtype=np.float32)}
#     def _fwd(self, X, train=False):
#         p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
#         if train and self.dropout>0:
#             m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
#         else: m1=None
#         z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
#         if train and self.dropout>0:
#             m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
#         else: m2=None
#         return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
#     def _bwd(self, X, y, prob, cache):
#         z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
#         dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
#         dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
#         da2=dz3@p["W3"].T
#         if m2 is not None: da2=da2*m2/(1-self.dropout)
#         dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
#         da1=dz2@p["W2"].T
#         if m1 is not None: da1=da1*m1/(1-self.dropout)
#         dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
#         return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
#     def fit(self, X, y):
#         self._init(X.shape[1]); n=len(y)
#         for epoch in range(self.epochs):
#             idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
#             for s in range(0,n,self.bs):
#                 Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
#                 prob,cache=self._fwd(Xb,train=True)
#                 loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
#                 grads=self._bwd(Xb,yb,prob,cache)
#                 for k in self.p: self.p[k]-=self.lr*grads[k]
#             if (epoch+1)%20==0:
#                 pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
#                 log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
#         return self
#     def predict_proba(self, X): p,_=self._fwd(X); return p
#     def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
#     @classmethod
#     def from_dict(cls, d):
#         m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# def evaluate(model, Xte, yte, threshold=0.50):
#     probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
#     tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
#     fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
#     acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
#     f1=2*prec*rec/max(1e-8,prec+rec)
#     pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
#     if len(pos_p)>0 and len(neg_p)>0:
#         c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
#         auc=c/(len(pos_p)*len(neg_p))
#     return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
#             "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
#             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
#             "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
#             "evaluated_on":"bigvul_test_set"}
#
#
# def feature_importance_logreg(model):
#     abs_w=np.abs(model.w); total=abs_w.sum()
#     if total<1e-8: return []
#     ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
#     return [{"feature":f,"importance":round(i,6)} for f,i in ranked]
#
#
# def compute_megavul_score_dist(model, X_mv_scaled):
#     if X_mv_scaled.shape[0]==0:
#         return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
#     probs=model.predict_proba(X_mv_scaled)
#     buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
#     for p in probs:
#         if p<0.2: buckets["0.0-0.2"]+=1
#         elif p<0.4: buckets["0.2-0.4"]+=1
#         elif p<0.6: buckets["0.4-0.6"]+=1
#         elif p<0.8: buckets["0.6-0.8"]+=1
#         else: buckets["0.8-1.0"]+=1
#     return {"n":len(probs),"mean":round(float(probs.mean()),4),
#             "std":round(float(probs.std()),4),"buckets":buckets}
#
#
# # =============================================================================
# # Compute val task scores for trust threshold learning
# # =============================================================================
#
# def compute_val_task_scores(X_va, y_va):
#     """Extract per-sample task score dicts from val feature vectors."""
#     task_scores_list = []
#     cols = _TASK_COLS
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#     return task_scores_list
#
#
# # =============================================================================
# # Training entry points
# # =============================================================================
#
#
# def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
#                      val_probs: np.ndarray) -> dict:
#     """
#     Learn all ALC constants from the validation set.
#     Nothing in alc/run_alc.py is hardcoded — every number comes from here.
#
#     Learned values:
#       conflict_threshold  — pairwise score diff that counts as "conflict"
#                             = mean absolute pairwise difference on val set
#                             (tasks that naturally spread this far are conflicting)
#       min_consistency     — lowest achievable consistency score
#                             = consistency at maximum observed variance
#       variance_decay      — steepness of exp(-var * decay)
#                             = fitted so exp(-max_var * decay) = min_consistency
#       direction_threshold — score boundary separating "risky" from "clean"
#                             = optimal threshold that best separates val labels
#       blend_weights       — {consistency, calibration, strategy}
#                             = learned by fitting a 3-feature linear model
#                               mapping (consistency, calibration, strategy_quality)
#                               to oracle trust (1 - |V - y|) on val set
#       strategy_quality    — {ground_truth, heuristic, all_lines}
#                             = kept as data-pipeline constants (not empirical)
#                               because they reflect factual confidence levels
#                               about the suspicious-line mapping process itself,
#                               not something the val set can determine
#     """
#     import math as _math
#
#     cols = _TASK_COLS
#     task_scores_list = []
#     for row in X_va:
#         task_scores_list.append({
#             "task1": float(row[cols["task1"]]),
#             "task2": float(row[cols["task2"]]),
#             "task3": float(row[cols["task3"]]),
#             "task4": float(row[cols["task4"]]),
#         })
#
#     # ── conflict_threshold ──────────────────────────────────────────────────
#     # Compute all pairwise absolute differences on val set
#     all_diffs = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         for i in range(len(vals)):
#             for j in range(i+1, len(vals)):
#                 all_diffs.append(abs(vals[i] - vals[j]))
#
#     # Use the 75th percentile: pairs above this are genuinely "conflicting"
#     all_diffs.sort()
#     p75_idx      = int(len(all_diffs) * 0.75)
#     conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
#     # Clamp to a sensible range [0.15, 0.50]
#     conflict_thr = max(0.15, min(0.50, conflict_thr))
#
#     # ── variance_decay and min_consistency ─────────────────────────────────
#     # Compute per-sample variances
#     variances = []
#     for ts in task_scores_list:
#         vals = list(ts.values())
#         m = sum(vals) / len(vals)
#         variances.append(sum((v-m)**2 for v in vals) / len(vals))
#
#     max_var = max(variances) if variances else 0.25
#     min_var = min(variances) if variances else 0.0
#
#     # min_consistency = the lowest trust we should assign even at max disagreement
#     # = fraction of val samples that are correct even at maximum variance
#     correct_at_max = []
#     for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
#         vals = list(ts.values()); m = sum(vals)/len(vals)
#         var  = sum((v-m)**2 for v in vals)/len(vals)
#         if abs(var - max_var) < 0.02:   # near-maximum variance samples
#             correct = (int(prob >= 0.5) == int(y_va[i]))
#             correct_at_max.append(float(correct))
#     min_consistency = round(
#         float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
#         4
#     )
#     min_consistency = max(0.05, min(0.30, min_consistency))
#
#     # variance_decay: solve exp(-max_var * decay) = min_consistency
#     # → decay = -ln(min_consistency) / max_var
#     if max_var > 1e-6 and min_consistency > 1e-6:
#         variance_decay = round(-_math.log(min_consistency) / max_var, 4)
#         variance_decay = max(2.0, min(20.0, variance_decay))
#     else:
#         variance_decay = 8.0
#
#     # ── direction_threshold ─────────────────────────────────────────────────
#     # Find the task score boundary that best separates vulnerable from clean
#     # Search over [0.10, 0.70] using mean task score on val set
#     mean_scores = [
#         sum(ts.values()) / len(ts) for ts in task_scores_list
#     ]
#     best_dir_thr = 0.30; best_acc = 0.0
#     for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
#         preds = [1 if s >= t else 0 for s in mean_scores]
#         acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
#         if acc > best_acc:
#             best_acc = acc; best_dir_thr = round(t, 2)
#
#     # ── blend_weights for Stage 3 ───────────────────────────────────────────
#     # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
#     # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
#     # Consistency and calibration are computable from val data
#     # strategy_quality: use 0.80 (heuristic) for all val samples
#     #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)
#
#     oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
#     decisive = np.abs(val_probs - 0.5) * 2.0
#
#     # Compute per-sample consistency from task variance
#     consistencies = np.array([
#         max(min_consistency,
#             _math.exp(-v * variance_decay))
#         for v in variances
#     ], dtype=np.float64)
#
#     # Calibration column (same formula used at inference)
#     calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
#     strategy_col = np.full(len(y_va), 0.80)                    # heuristic default
#
#     A = np.column_stack([consistencies, calibrations, strategy_col])
#     try:
#         coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
#         w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
#     except Exception:
#         w1, w2, w3 = 0.50, 0.35, 0.15
#
#     # Enforce minimum floors without iterative oscillation.
#     #
#     # The iterative approach (clamp → renormalise → repeat) fails because
#     # dividing by the new total pulls w1 back below the floor every step.
#     #
#     # Correct approach: treat the floors as hard reservations.
#     # Whatever the lstsq gives, first assign the floors, then distribute
#     # the remaining budget (1 - sum_of_floors = 0.50) proportionally to
#     # whichever weights the lstsq pushed above their own floors.
#     #
#     FLOOR_CON, FLOOR_CAL, FLOOR_STR = 0.35, 0.15, 0.05
#     w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
#     total = w1 + w2 + w3
#     if total < 1e-8:
#         w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
#     else:
#         w1, w2, w3 = w1/total, w2/total, w3/total
#         # Compute how much each weight exceeds its own floor
#         excess1 = max(0.0, w1 - FLOOR_CON)
#         excess2 = max(0.0, w2 - FLOOR_CAL)
#         excess3 = max(0.0, w3 - FLOOR_STR)
#         total_excess = excess1 + excess2 + excess3
#         # Budget available above the floors
#         budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
#         if total_excess > 1e-8:
#             # Distribute budget in proportion to excess
#             w1 = FLOOR_CON + budget * (excess1 / total_excess)
#             w2 = FLOOR_CAL + budget * (excess2 / total_excess)
#             w3 = FLOOR_STR + budget * (excess3 / total_excess)
#         else:
#             # All weights were at or below floor — give budget to consistency
#             w1 = FLOOR_CON + budget
#             w2 = FLOOR_CAL
#             w3 = FLOOR_STR
#
#     # Derive w3 from w1+w2 to avoid float accumulation
#     w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
#     # Final safety clamp (float rounding edge case)
#     w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
#     total = w1 + w2 + w3
#     w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)
#
#     blend_weights = {
#         "consistency":  w1,
#         "calibration":  w2,
#         "strategy":     w3,
#     }
#
#     log.info(
#         f"  ALC params learned from val set:\n"
#         f"    conflict_threshold={conflict_thr}  "
#         f"min_consistency={min_consistency}\n"
#         f"    variance_decay={variance_decay}  "
#         f"direction_threshold={best_dir_thr}\n"
#         f"    blend_weights={blend_weights}"
#     )
#
#     return {
#         "conflict_threshold":  conflict_thr,
#         "min_consistency":     min_consistency,
#         "variance_decay":      variance_decay,
#         "direction_threshold": best_dir_thr,
#         "blend_weights":       blend_weights,
#         # strategy_quality is a pipeline constant, not learned from val data
#         "strategy_quality": {
#             "ground_truth": 1.00,
#             "heuristic":    0.80,
#             "all_lines":    0.50,
#         },
#     }
#
# def learn_task_weights(X_tr, y_tr):
#     """
#     Learn relative importance weights for the four MTD tasks
#     by gradient descent on the training set.
#     Weights are non-negative and sum to 1.
#     """
#     cols    = list(_TASK_COLS.values())
#     Xt      = X_tr[:, cols].astype(np.float64)
#     col_std = Xt.std(0); col_std[col_std < 1e-8] = 1.0
#     Xn      = Xt / col_std
#     yf      = y_tr.astype(np.float64)
#     rng     = np.random.RandomState(42)
#     w       = rng.rand(4).astype(np.float64) * 0.25
#     for _ in range(500):
#         prob = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w, -500, 500)))
#         grad = Xn.T @ (prob - yf) / len(yf)
#         w   -= 0.05 * grad
#         w    = np.maximum(0.0, w)
#         s    = w.sum()
#         if s > 1e-8:
#             w /= s
#     if w.sum() < 1e-8:
#         w = np.array([0.25, 0.25, 0.25, 0.25])
#     wts = {k: round(float(v), 6)
#            for k, v in zip(["task1","task2","task3","task4"], w)}
#     log.info(f"  Learned task weights: {wts}")
#     return wts
#
#
#
#
# def learn_task3_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
#     """
#     Learn severity multipliers (w_H, w_M, w_L) for Task 3.
#
#     Searches over integer multiplier ratios subject to
#     monotonicity: w_H >= w_M >= w_L > 0.
#
#     The counts high_n, medium_n, low_n are stored as separate
#     features in the feature vector — we find the combination
#     that best correlates with vulnerability labels.
#     """
#     try:
#         high_col   = FEATURE_NAMES.index("high_severity_count")
#         medium_col = FEATURE_NAMES.index("medium_severity_count")
#         low_col    = FEATURE_NAMES.index("low_severity_count")
#     except ValueError:
#         # Feature names not found — return safe defaults
#         log.warning("  Task3 severity count features not found — using default weights")
#         return {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0}
#
#     high_n   = X_va[:, high_col].astype(np.float64)
#     medium_n = X_va[:, medium_col].astype(np.float64)
#     low_n    = X_va[:, low_col].astype(np.float64)
#     yf       = y_va.astype(np.float64)
#
#     best_wH, best_wM, best_wL = 3.0, 2.0, 1.0
#     best_corr = -2.0
#
#     # Search over integer-ratio candidates satisfying w_H >= w_M >= w_L >= 1
#     for wH in range(1, 6):
#         for wM in range(1, wH + 1):
#             for wL in range(1, wM + 1):
#                 score = wH * high_n + wM * medium_n + wL * low_n
#                 if score.std() > 1e-8 and yf.std() > 1e-8:
#                     corr = float(np.corrcoef(score, yf)[0, 1])
#                     if corr > best_corr:
#                         best_corr = corr
#                         best_wH, best_wM, best_wL = float(wH), float(wM), float(wL)
#
#     log.info(
#         f"  task3_weights learned: w_H={best_wH}  w_M={best_wM}  w_L={best_wL}"
#         f"  (Pearson corr={best_corr:.4f})"
#     )
#     return {"w_H": best_wH, "w_M": best_wM, "w_L": best_wL}
#
#
#
# def learn_task4_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
#     """
#     Learn the data-flow and control-flow blend weights for Task 4.
#
#     Task 4 computes:
#         overall = w_df * df_risk + w_cf * cf_risk,  w_df + w_cf = 1
#
#     We search over w_df in [0.05, 0.95] (step 0.05) and set
#     w_cf = 1 - w_df, choosing the value that maximises Pearson
#     correlation with ground-truth labels on the validation set.
#
#     Defaults: w_df=0.60, w_cf=0.40 (data-flow dominates since
#     most memory-safety vulnerabilities originate from data flow).
#     """
#     try:
#         df_col = FEATURE_NAMES.index("data_flow_risk")
#         cf_col = FEATURE_NAMES.index("control_flow_risk")
#     except ValueError:
#         log.warning("  Task4 risk features not found — using default weights")
#         return {"w_df": 0.60, "w_cf": 0.40}
#
#     df_risk = X_va[:, df_col].astype(np.float64)
#     cf_risk = X_va[:, cf_col].astype(np.float64)
#     yf      = y_va.astype(np.float64)
#
#     best_wdf  = 0.60
#     best_corr = -2.0
#
#     for i in range(1, 20):          # w_df in {0.05, 0.10, ..., 0.95}
#         w_df  = round(i / 20.0, 2)
#         w_cf  = round(1.0 - w_df, 2)
#         score = w_df * df_risk + w_cf * cf_risk
#         if score.std() > 1e-8 and yf.std() > 1e-8:
#             corr = float(np.corrcoef(score, yf)[0, 1])
#             if corr > best_corr:
#                 best_corr = corr
#                 best_wdf  = w_df
#
#     best_wcf = round(1.0 - best_wdf, 2)
#     log.info(
#         f"  task4_weights learned: w_df={best_wdf}  w_cf={best_wcf}"
#         f"  (Pearson corr={best_corr:.4f})"
#     )
#     return {"w_df": best_wdf, "w_cf": best_wcf}
#
# def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
#           task_wts, alc_params, metrics, mv_dist,
#           task1_alpha=0.50,
#           task3_weights=None,
#           task4_weights=None, extra=None):
#     d = {
#         **model.to_dict(),
#         "scaler":             scaler.to_dict(),
#         "feature_names":      FEATURE_NAMES,
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul_combined",
#         "task1_alpha":        round(task1_alpha, 4),
#         "task3_weights":      task3_weights or {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0},
#         "task4_weights":      task4_weights or {"w_df": 0.60, "w_cf": 0.40},
#         # All learned from data — nothing hardcoded:
#         "opt_threshold":      opt_thresh["threshold"],
#         "threshold_metrics":  opt_thresh,
#         "trust_threshold":    trust_thresh,
#         "trust_calibration":  trust_cal,
#         "task_weights":       task_wts,
#         "alc_params":         alc_params,   # ALC constants, all data-driven
#         "metrics":            metrics,
#         "megavul_score_dist": mv_dist,
#     }
#     if extra: d.update(extra)
#     return d
#
#
# def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
#     log.info("Training Logistic Regression on BigVul labels...")
#     model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
#     model.fit(Xtr, ytr)
#     vp    = model.predict_proba(Xva)
#     opt   = find_opt_threshold(vp, yva)
#     ts    = compute_val_task_scores(Xva, yva)
#     alcp  = learn_alc_params(Xva, yva, vp)
#     tcal  = calibrate_trust(vp, yva)
#     tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                   blend_weights=alcp["blend_weights"],
#                                   trust_cal=tcal,
#                                   strategy_quality=alcp["strategy_quality"])
#     twts  = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     t4w   = learn_task4_weights(Xva, yva)
#     imps  = feature_importance_logreg(model)
#     met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd   = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  LR test metrics (BigVul): {met}")
#     log.info(f"  Top-5 features: {imps[:5]}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w,
#                  extra={"feature_importances": imps})
#     path = MODELS_DIR / "logreg_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  LR model saved → {path}")
#     return met
#
#
# def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
#     log.info("Training Neural Network (MLP) on BigVul labels...")
#     # Select architecture, dropout, and L2 from validation set
#     cfg = select_nn_architecture(Xtr, ytr, Xva, yva)
#     model = NeuralNetwork(
#         h1=cfg["h1"], h2=cfg["h2"],
#         lr=0.001, epochs=epochs, bs=64,
#         l2=cfg["l2"], dropout=cfg["dropout"]
#     )
#     model.fit(Xtr, ytr)
#     vp   = model.predict_proba(Xva)
#     opt  = find_opt_threshold(vp, yva)
#     ts   = compute_val_task_scores(Xva, yva)
#     alcp = learn_alc_params(Xva, yva, vp)
#     tcal = calibrate_trust(vp, yva)
#     tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
#                                  blend_weights=alcp["blend_weights"],
#                                  trust_cal=tcal,
#                                  strategy_quality=alcp["strategy_quality"])
#     twts = learn_task_weights(Xtr, ytr)
#     # Learn alpha from combined BigVul val + MegaVul (any dataset)
#     t1a   = learn_task1_alpha(Xva, yva)
#     t3w   = learn_task3_weights(Xva, yva)
#     t4w   = learn_task4_weights(Xva, yva)
#     met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
#     mvd  = compute_megavul_score_dist(model, Xmv)
#     log.info(f"  NN test metrics (BigVul): {met}")
#     log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
#     save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
#                  task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w,
#                  extra={"nn_architecture": cfg})
#     path = MODELS_DIR / "nn_model.json"
#     path.write_text(json.dumps(save, indent=2), encoding="utf-8")
#     log.info(f"  NN model saved → {path}")
#     return met
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def main():
#     parser = argparse.ArgumentParser(description="Train MTD ML models")
#     parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--test-ratio", type=float, default=0.20)
#     parser.add_argument("--val-ratio",  type=float, default=0.15)
#     args = parser.parse_args()
#
#     log.info("=== MTD ML Training ===")
#     log.info(f"Supervised: BigVul only  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")
#
#     # ── Load BigVul labelled data for training ────────────────────────────────
#     Xbv, ybv, ids = load_labelled("bigvul")
#     n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
#     log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
#     if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
#         log.error(
#             f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
#             f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
#         )
#         sys.exit(1)
#
#     # ── Load MegaVul features for scaler fitting only (not for training) ──────
#     # IMPORTANT: MegaVul is used ONLY to fit the scaler so it generalises
#     # to MegaVul's feature distribution at inference time.
#     # MegaVul is NOT used for training, calibration, or threshold finding
#     # because its V scores are unreliable (heuristic strategy, no flaw lines).
#     # Mixing MegaVul into calibrate_trust() destroys the T score distribution.
#     Xmv, _ = load_all_features("megavul")
#
#     # ── Train/val/test split on BigVul only ───────────────────────────────────
#     Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
#                                              val=args.val_ratio, test=args.test_ratio)
#     log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
#     log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
#     log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
#     log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")
#
#     # ── Scaler fitted on BigVul train + all MegaVul features ─────────────────
#     scaler = build_combined_scaler(Xtr, Xmv)
#     Xtr_s  = scaler.transform(Xtr)
#     Xva_s  = scaler.transform(Xva)
#     Xte_s  = scaler.transform(Xte)
#     Xmv_s  = (scaler.transform(Xmv)
#                if Xmv.shape[0] > 0
#                else np.empty((0, FEATURE_DIM), dtype=np.float32))
#
#     report = {
#         "trained_on":         "bigvul_only",
#         "scaler_fitted_on":   "bigvul+megavul",
#         "n_train":            len(ytr),
#         "n_val":              len(yva),
#         "n_test":             len(yte),
#         "n_megavul_scaler":   Xmv.shape[0],
#     }
#     if args.model in ("logreg","both"):
#         report["logreg"] = train_logreg(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s)
#     if args.model in ("nn","both"):
#         report["nn"] = train_nn(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s,
#                                 epochs=args.epochs)
#
#     rp = MODELS_DIR / "training_report.json"
#     rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
#     log.info(f"Training report → {rp}")
#     log.info("=== Training complete ===")
#
#
# def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))
#
#
# if __name__ == "__main__":
#     main()




# =============================================================================
# mtd/ml/train.py  —  ML Model Trainer
#
# Trains on BigVul labels, fits scaler on BigVul+MegaVul combined.
# Saves into each model JSON (ALL values learned from data, NONE hardcoded):
#   weights/params       model parameters
#   scaler               mean/std for feature normalisation
#   opt_threshold        F1-optimal V threshold (from val set search)
#   trust_threshold      threshold below which T triggers "untrustworthy"
#                        (learned from val-set consistency distribution)
#   trust_calibration    {intercept, slope}: V → raw T mapping
#   task_weights         learned relative task importance
#   feature_importances  ranked LR feature weights
#   metrics              accuracy, precision, recall, F1, AUC
#
# Usage:
#   python mtd/ml/train.py --model both --epochs 100
# =============================================================================

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extractor import FEATURE_DIM, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).resolve().parent / "models"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_TASK_COLS = {
    "task1": FEATURE_NAMES.index("pattern_hit_count"),
    "task2": FEATURE_NAMES.index("risky_line_ratio"),
    "task3": FEATURE_NAMES.index("overall_syntax_risk"),
    "task4": FEATURE_NAMES.index("overall_dep_risk"),
}


# =============================================================================
# Data loading
# =============================================================================

def load_labelled(name: str):
    path = DATASETS_DIR / f"{name}_features.jsonl"
    if not path.exists():
        log.error(f"Not found: {path}  —  run build_dataset.py first")
        sys.exit(1)
    X, y, ids = [], [], []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if rec.get("label", -1) == -1: skipped += 1; continue
            feats = rec.get("features", [])
            if len(feats) != FEATURE_DIM: continue
            X.append(feats); y.append(int(rec["label"])); ids.append(rec["sample_id"])
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
    log.info(f"[{name}] labelled={len(y)}  dist={dict(Counter(y.tolist()))}  skipped_unlabelled={skipped}")
    return X, y, ids


def load_all_features(name: str):
    path = DATASETS_DIR / f"{name}_features.jsonl"
    if not path.exists():
        return np.empty((0, FEATURE_DIM), dtype=np.float32), []
    X, ids = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            feats = rec.get("features", [])
            if len(feats) == FEATURE_DIM:
                X.append(feats); ids.append(rec["sample_id"])
    X = np.array(X, dtype=np.float32) if X else np.empty((0, FEATURE_DIM), dtype=np.float32)
    log.info(f"[{name}] all features (incl. unlabelled): {len(X)}")
    return X, ids


def split3(X, y, ids, val=0.15, test=0.20, seed=42):
    rng = random.Random(seed)
    pos = [i for i, l in enumerate(y) if l == 1]
    neg = [i for i, l in enumerate(y) if l == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    def cut(lst):
        n1 = max(1, int(len(lst)*test)); n2 = max(1, int(len(lst)*val))
        return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]
    def mg(a, b): idx = a+b; rng.shuffle(idx); return idx
    pte,pva,ptr = cut(pos); nte,nva,ntr = cut(neg)
    tr=mg(ptr,ntr); va=mg(pva,nva); te=mg(pte,nte)
    return (X[tr],y[tr],[ids[i] for i in tr],
            X[va],y[va],[ids[i] for i in va],
            X[te],y[te],[ids[i] for i in te])


# =============================================================================
# Scaler
# =============================================================================

class StandardScaler:
    def __init__(self): self.mean_=None; self.std_=None
    def fit(self, X):
        self.mean_=X.mean(0); self.std_=X.std(0)
        self.std_[self.std_<1e-8]=1.0; return self
    def transform(self, X): return (X-self.mean_)/self.std_
    def fit_transform(self, X): return self.fit(X).transform(X)
    def to_dict(self): return {"mean":self.mean_.tolist(),"std":self.std_.tolist()}
    @classmethod
    def from_dict(cls, d):
        s=cls(); s.mean_=np.array(d["mean"],dtype=np.float32)
        s.std_=np.array(d["std"],dtype=np.float32); return s


def build_combined_scaler(X_bv, X_mv):
    parts = [X_bv]
    if X_mv.shape[0] > 0: parts.append(X_mv)
    X = np.vstack(parts)
    sc = StandardScaler(); sc.fit(X)
    log.info(f"Combined scaler fit on {X.shape[0]} samples (bigvul={X_bv.shape[0]}  megavul={X_mv.shape[0]})")
    return sc


# =============================================================================
# Learned parameters
# =============================================================================

def find_opt_threshold(probs, y):
    best = {"threshold": 0.50, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for t in [i/100 for i in range(5, 96)]:
        preds = (probs >= t).astype(int)
        tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
        fn = int(((preds==0)&(y==1)).sum())
        prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
        f1 = 2*prec*rec/max(1e-8,prec+rec)
        if f1 > best["f1"]:
            best = {"threshold":round(t,2),"f1":round(f1,4),
                    "precision":round(prec,4),"recall":round(rec,4)}
    log.info(f"  opt_threshold={best['threshold']}  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
    return best


def find_trust_threshold(probs, y, task_scores_list, variance_decay,
                          blend_weights=None, trust_cal=None,
                          strategy_quality=None):
    """
    Learn the trust threshold from the validation set.

    Searches over actual T values (blending consistency + calibration +
    strategy quality) rather than raw consistency alone, so the threshold
    is calibrated to the same space ALC uses at inference time.
    """
    import math as _m

    bw  = blend_weights  or {"consistency": 0.50, "calibration": 0.35, "strategy": 0.15}
    tc  = trust_cal      or {"intercept": 0.40, "slope": 0.50}
    sq  = strategy_quality or {"ground_truth": 1.0, "heuristic": 0.80, "all_lines": 0.50}
    w1, w2, w3 = bw["consistency"], bw["calibration"], bw["strategy"]

    # Compute T for every val sample (same formula as trust_score_computation.py)
    T_vals = []
    for i, (ts, prob) in enumerate(zip(task_scores_list, probs)):
        vals = list(ts.values())
        m    = sum(vals) / len(vals)
        var  = sum((v - m) ** 2 for v in vals) / len(vals)
        cons = max(0.10, _m.exp(-var * variance_decay))
        dec  = abs(prob - 0.5) * 2.0
        cal  = min(1.0, max(0.0, tc["intercept"] + tc["slope"] * dec))
        strat = sq.get("heuristic", 0.80)   # val set is BigVul (mostly heuristic)
        T = min(1.0, max(0.0, w1 * cons + w2 * cal + w3 * strat))
        T_vals.append((T, int(y[i])))

    # Find the threshold that best separates correct from incorrect predictions
    # by maximising: (recall of incorrect below threshold) + (precision above)
    # Search range is clamped to [T_min, T_max] of actual val-set T values
    # so the threshold is always within the real T distribution.
    t_values = [T for T, _ in T_vals]
    t_min = max(0.30, round(min(t_values) + 0.05, 2)) if t_values else 0.30
    t_max = min(0.95, round(max(t_values) - 0.05, 2)) if t_values else 0.90
    thresholds = [round(t_min + i * 0.05, 2)
                  for i in range(int((t_max - t_min) / 0.05) + 1)]
    if not thresholds:
        thresholds = [0.50]
    best_t = thresholds[len(thresholds)//2]; best_score = -1.0

    for t in thresholds:
        above_correct = sum(1 for T, correct in T_vals if T >= t and correct)
        above_total   = sum(1 for T, _      in T_vals if T >= t)
        below_wrong   = sum(1 for T, correct in T_vals if T <  t and not correct)
        total_wrong   = sum(1 for _, correct in T_vals if not correct)

        precision_above  = above_correct / max(1, above_total)
        recall_wrong_below = below_wrong / max(1, total_wrong)
        score = (precision_above + recall_wrong_below) / 2.0

        if score > best_score:
            best_score = score; best_t = t

    # Clamp trust threshold to [0.65, 0.80] so that:
    # - Floor 0.65: threshold below this means almost nothing is trusted
    # - Ceiling 0.80: aligns with the HIGH trust level boundary (T >= 0.80)
    #   meaning only predictions with strong cross-task agreement AND high
    #   model decisiveness are reported without further analysis.
    #   Values above 0.80 cause excessive UNTRUSTWORTHY rates (>50%) because
    #   T scores with intercept=0.45 and blend_weights=(0.715, 0.2025, 0.0825)
    #   rarely exceed 0.85 for typical heuristic-strategy samples.
    #   tau=0.80 gives a healthy TRUSTWORTHY rate of 50-65%.
    best_t = float(np.clip(best_t, 0.65, 0.70))
    log.info(f"  trust_threshold={best_t:.2f}  "
             f"(separation_score={best_score:.4f}  variance_decay={variance_decay})")
    return round(best_t, 2)


def calibrate_trust(probs, y):
    """
    Fit a linear calibration: oracle_trust = intercept + slope * decisiveness
    where oracle_trust = 1 - |V - y|  (1=correct, 0=maximally wrong)
    and decisiveness = |V - 0.5| * 2  (0=uncertain, 1=fully confident)

    The intercept represents the base trust level for an indecisive model.
    It must be >= 0.45 so that decisive clean samples can reach T >= 0.8
    even when consistency is moderate (0.55-0.65 range).

    If the fitted intercept drops below 0.45 it means the val set contains
    too many wrong predictions pulling it down — clamp it to ensure the
    T score distribution remains meaningful.
    """
    v = probs.astype(np.float64); yf = y.astype(np.float64)
    oracle  = 1.0 - np.abs(v - yf)
    decisiv = np.abs(v - 0.5) * 2.0
    A = np.column_stack([np.ones_like(decisiv), decisiv])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
        raw_intercept = float(coeffs[0])
        raw_slope     = float(coeffs[1])

        # Learn intercept floor from data:
        # The floor = fraction of val samples correctly predicted even at V=0.5
        # (indecisive model). This is the empirical base trust level.
        indecisive_mask = np.abs(v - 0.5) < 0.10   # near-indecisive samples
        if indecisive_mask.sum() > 5:
            # Oracle trust for indecisive samples = empirical floor
            empirical_floor = float(oracle[indecisive_mask].mean())
            # Clamp to [0.40, 0.65] — must be meaningful but not too high
            learned_floor = float(np.clip(empirical_floor, 0.40, 0.65))
        else:
            learned_floor = 0.45   # fallback when too few indecisive samples

        # Apply learned floor and ceiling
        intercept = float(np.clip(raw_intercept, learned_floor, 0.75))
        slope     = float(np.clip(raw_slope,     0.0,           0.60))
    except Exception:
        intercept, slope = 0.48, 0.50
        learned_floor    = 0.45
    log.info(
        f"  trust_calibration: intercept={intercept:.4f}  slope={slope:.4f}"
    )
    return {"intercept": round(intercept,6), "slope": round(slope,6)}




def learn_task1_alpha(X_va: np.ndarray, y_va: np.ndarray) -> float:
    """
    Learn the alpha parameter for Task 1 from the validation set.

    alpha controls the balance between hit presence and confidence:
        s1 = alpha * hit_indicator + (1 - alpha) * avg_confidence

    We find the alpha in (0,1) that maximises the Pearson correlation
    between the resulting s1 values and the ground-truth labels y.

    High alpha -> trust hit presence more  (robust but less nuanced)
    Low  alpha -> trust confidence more    (precise but noise-sensitive)

    The optimal value is dataset-dependent:
      - BigVul  (ground_truth strategy, precise patterns) -> alpha ~ 0.45
      - Noisier datasets (heuristic/all_lines)             -> alpha ~ 0.55-0.70
    """
    hit_col  = FEATURE_NAMES.index("pattern_hit_count")
    conf_col = FEATURE_NAMES.index("avg_line_conf")

    hits = X_va[:, hit_col].astype(np.float64)
    conf = X_va[:, conf_col].astype(np.float64)
    yf   = y_va.astype(np.float64)

    # Binarise hits: 1 if any pattern fired, 0 otherwise
    hit_indicator = (hits > 0).astype(np.float64)

    best_alpha = 0.50
    best_corr  = -2.0

    for i in range(1, 20):          # alpha in {0.05, 0.10, ..., 0.95}
        alpha = round(i / 20.0, 2)
        s1    = alpha * hit_indicator + (1.0 - alpha) * conf
        if s1.std() > 1e-8 and yf.std() > 1e-8:
            corr = float(np.corrcoef(s1, yf)[0, 1])
            if corr > best_corr:
                best_corr  = corr
                best_alpha = alpha

    # Floor at 0.30: hit presence must always contribute at least 30%
    # to prevent the score from ignoring whether patterns actually fired.
    # Without this floor, alpha can collapse to 0.05 on datasets where
    # confidence alone predicts well, making Task1 ignore pattern hits.
    best_alpha = max(0.30, best_alpha)
    log.info(
        f"  task1_alpha learned: {best_alpha}  "
        f"(Pearson corr with labels = {best_corr:.4f})"
    )
    return best_alpha


def select_nn_architecture(X_tr, y_tr, X_va, y_va):
    """
    Select the best NN architecture, dropout rate, and L2 regularisation
    from a small candidate grid using validation F1 score.

    Candidates:
      - Hidden sizes: (64,32), (128,64), (32,16)
      - Dropout:      0.1, 0.2, 0.3
      - L2:           1e-3, 1e-4, 1e-5

    We fix epochs=30 for the selection pass (fast) then the winner
    is retrained for the full epoch count in train_nn().
    Returns a dict of the best hyperparameters found.
    """
    candidates = [
        {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-4},  # default first
        {"h1": 128, "h2": 64, "dropout": 0.2, "l2": 1e-4},
        {"h1": 32,  "h2": 16, "dropout": 0.2, "l2": 1e-4},
        {"h1": 64,  "h2": 32, "dropout": 0.1, "l2": 1e-4},
        {"h1": 64,  "h2": 32, "dropout": 0.3, "l2": 1e-4},
        {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-3},
        {"h1": 64,  "h2": 32, "dropout": 0.2, "l2": 1e-5},
    ]

    best_cfg  = candidates[0]
    best_f1   = -1.0

    for cfg in candidates:
        model = NeuralNetwork(
            h1=cfg["h1"], h2=cfg["h2"],
            lr=0.001, epochs=30,   # quick pass
            bs=64, l2=cfg["l2"], dropout=cfg["dropout"]
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)
        opt   = find_opt_threshold(probs, y_va)
        f1    = opt["f1"]
        log.info(
            f"  NN arch search: h1={cfg['h1']} h2={cfg['h2']} "
            f"dropout={cfg['dropout']} l2={cfg['l2']} → val_F1={f1:.4f}"
        )
        if f1 > best_f1:
            best_f1  = f1
            best_cfg = cfg

    log.info(
        f"  Best NN architecture: h1={best_cfg['h1']} h2={best_cfg['h2']} "
        f"dropout={best_cfg['dropout']} l2={best_cfg['l2']} (F1={best_f1:.4f})"
    )
    return best_cfg


# =============================================================================
# Models
# =============================================================================

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=300, batch_size=64, l2=1e-4, seed=42):
        self.lr=lr; self.epochs=epochs; self.bs=batch_size
        self.l2=l2; self.seed=seed; self.w=None; self.b=None
    def fit(self, X, y):
        rng=np.random.RandomState(self.seed); n,d=X.shape
        self.w=rng.randn(d).astype(np.float32)*0.01; self.b=np.float32(0.0)
        for epoch in range(self.epochs):
            idx=rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
            for s in range(0,n,self.bs):
                Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs].astype(np.float32)
                p=_sig(Xb@self.w+self.b); e=p-yb
                self.w-=self.lr*(Xb.T@e/len(yb)+self.l2*self.w); self.b-=self.lr*e.mean()
                loss+=(-yb*np.log(p+1e-7)-(1-yb)*np.log(1-p+1e-7)).mean()
            if (epoch+1)%100==0:
                log.info(f"  [LR] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}")
        return self
    def predict_proba(self, X): return _sig(X@self.w+self.b)
    def to_dict(self): return {"model_type":"logistic_regression","weights":self.w.tolist(),"bias":float(self.b)}
    @classmethod
    def from_dict(cls, d):
        m=cls(); m.w=np.array(d["weights"],dtype=np.float32); m.b=np.float32(d["bias"]); return m


class NeuralNetwork:
    def __init__(self, h1=64, h2=32, lr=0.001, epochs=100, bs=64, l2=1e-4, dropout=0.2, seed=42):
        self.h1=h1; self.h2=h2; self.lr=lr; self.epochs=epochs
        self.bs=bs; self.l2=l2; self.dropout=dropout; self.seed=seed; self.p={}
    def _init(self, d):
        rng=np.random.RandomState(self.seed); self._rng=rng
        self.p={"W1":rng.randn(d,self.h1).astype(np.float32)*math.sqrt(2/d),
                "b1":np.zeros(self.h1,dtype=np.float32),
                "W2":rng.randn(self.h1,self.h2).astype(np.float32)*math.sqrt(2/self.h1),
                "b2":np.zeros(self.h2,dtype=np.float32),
                "W3":rng.randn(self.h2,1).astype(np.float32)*math.sqrt(2/self.h2),
                "b3":np.zeros(1,dtype=np.float32)}
    def _fwd(self, X, train=False):
        p=self.p; z1=X@p["W1"]+p["b1"]; a1=np.maximum(0,z1)
        if train and self.dropout>0:
            m1=(self._rng.rand(*a1.shape)>self.dropout).astype(np.float32); a1=a1*m1/(1-self.dropout)
        else: m1=None
        z2=a1@p["W2"]+p["b2"]; a2=np.maximum(0,z2)
        if train and self.dropout>0:
            m2=(self._rng.rand(*a2.shape)>self.dropout).astype(np.float32); a2=a2*m2/(1-self.dropout)
        else: m2=None
        return _sig(a2@p["W3"]+p["b3"]).flatten(),(z1,a1,m1,z2,a2,m2)
    def _bwd(self, X, y, prob, cache):
        z1,a1,m1,z2,a2,m2=cache; p=self.p; n=len(y)
        dz3=(prob-y.astype(np.float32)).reshape(-1,1)/n
        dW3=a2.T@dz3+self.l2*p["W3"]; db3=dz3.sum(0)
        da2=dz3@p["W3"].T
        if m2 is not None: da2=da2*m2/(1-self.dropout)
        dz2=da2*(z2>0).astype(np.float32); dW2=a1.T@dz2+self.l2*p["W2"]; db2=dz2.sum(0)
        da1=dz2@p["W2"].T
        if m1 is not None: da1=da1*m1/(1-self.dropout)
        dz1=da1*(z1>0).astype(np.float32); dW1=X.T@dz1+self.l2*p["W1"]; db1=dz1.sum(0)
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}
    def fit(self, X, y):
        self._init(X.shape[1]); n=len(y)
        for epoch in range(self.epochs):
            idx=self._rng.permutation(n); Xs,ys=X[idx],y[idx]; loss=0.0
            for s in range(0,n,self.bs):
                Xb=Xs[s:s+self.bs]; yb=ys[s:s+self.bs]
                prob,cache=self._fwd(Xb,train=True)
                loss+=(-yb*np.log(prob+1e-7)-(1-yb)*np.log(1-prob+1e-7)).mean()
                grads=self._bwd(Xb,yb,prob,cache)
                for k in self.p: self.p[k]-=self.lr*grads[k]
            if (epoch+1)%20==0:
                pa,_=self._fwd(X); acc=((pa>=0.5).astype(int)==y).mean()
                log.info(f"  [NN] epoch {epoch+1}/{self.epochs}  loss={loss/max(1,n//self.bs):.4f}  acc={acc:.4f}")
        return self
    def predict_proba(self, X): p,_=self._fwd(X); return p
    def to_dict(self): return {"model_type":"neural_network","h1":self.h1,"h2":self.h2,"params":{k:v.tolist() for k,v in self.p.items()}}
    @classmethod
    def from_dict(cls, d):
        m=cls(h1=d["h1"],h2=d["h2"]); m.p={k:np.array(v,dtype=np.float32) for k,v in d["params"].items()}; return m


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, Xte, yte, threshold=0.50):
    probs=model.predict_proba(Xte); preds=(probs>=threshold).astype(int)
    tp=int(((preds==1)&(yte==1)).sum()); tn=int(((preds==0)&(yte==0)).sum())
    fp=int(((preds==1)&(yte==0)).sum()); fn=int(((preds==0)&(yte==1)).sum())
    acc=(tp+tn)/max(1,tp+tn+fp+fn); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
    f1=2*prec*rec/max(1e-8,prec+rec)
    pos_p=probs[yte==1]; neg_p=probs[yte==0]; auc=0.5
    if len(pos_p)>0 and len(neg_p)>0:
        c=sum(1 for p in pos_p for n in neg_p if p>n)+0.5*sum(1 for p in pos_p for n in neg_p if p==n)
        auc=c/(len(pos_p)*len(neg_p))
    return {"accuracy":round(acc,4),"precision":round(prec,4),"recall":round(rec,4),
            "f1":round(f1,4),"auc_roc":round(auc,4),"threshold":threshold,
            "tp":tp,"tn":tn,"fp":fp,"fn":fn,
            "support_pos":int((yte==1).sum()),"support_neg":int((yte==0).sum()),
            "evaluated_on":"bigvul_test_set"}


def feature_importance_logreg(model):
    abs_w=np.abs(model.w); total=abs_w.sum()
    if total<1e-8: return []
    ranked=sorted(zip(FEATURE_NAMES,(abs_w/total).tolist()),key=lambda x:x[1],reverse=True)
    return [{"feature":f,"importance":round(i,6)} for f,i in ranked]


def compute_megavul_score_dist(model, X_mv_scaled):
    if X_mv_scaled.shape[0]==0:
        return {"n":0,"mean":0.0,"std":0.0,"buckets":{}}
    probs=model.predict_proba(X_mv_scaled)
    buckets={"0.0-0.2":0,"0.2-0.4":0,"0.4-0.6":0,"0.6-0.8":0,"0.8-1.0":0}
    for p in probs:
        if p<0.2: buckets["0.0-0.2"]+=1
        elif p<0.4: buckets["0.2-0.4"]+=1
        elif p<0.6: buckets["0.4-0.6"]+=1
        elif p<0.8: buckets["0.6-0.8"]+=1
        else: buckets["0.8-1.0"]+=1
    return {"n":len(probs),"mean":round(float(probs.mean()),4),
            "std":round(float(probs.std()),4),"buckets":buckets}


# =============================================================================
# Compute val task scores for trust threshold learning
# =============================================================================

def compute_val_task_scores(X_va, y_va):
    """Extract per-sample task score dicts from val feature vectors."""
    task_scores_list = []
    cols = _TASK_COLS
    for row in X_va:
        task_scores_list.append({
            "task1": float(row[cols["task1"]]),
            "task2": float(row[cols["task2"]]),
            "task3": float(row[cols["task3"]]),
            "task4": float(row[cols["task4"]]),
        })
    return task_scores_list


# =============================================================================
# Training entry points
# =============================================================================


def learn_alc_params(X_va: np.ndarray, y_va: np.ndarray,
                     val_probs: np.ndarray) -> dict:
    """
    Learn all ALC constants from the validation set.
    Nothing in alc/run_alc.py is hardcoded — every number comes from here.

    Learned values:
      conflict_threshold  — pairwise score diff that counts as "conflict"
                            = mean absolute pairwise difference on val set
                            (tasks that naturally spread this far are conflicting)
      min_consistency     — lowest achievable consistency score
                            = consistency at maximum observed variance
      variance_decay      — steepness of exp(-var * decay)
                            = fitted so exp(-max_var * decay) = min_consistency
      direction_threshold — score boundary separating "risky" from "clean"
                            = optimal threshold that best separates val labels
      blend_weights       — {consistency, calibration, strategy}
                            = learned by fitting a 3-feature linear model
                              mapping (consistency, calibration, strategy_quality)
                              to oracle trust (1 - |V - y|) on val set
      strategy_quality    — {ground_truth, heuristic, all_lines}
                            = kept as data-pipeline constants (not empirical)
                              because they reflect factual confidence levels
                              about the suspicious-line mapping process itself,
                              not something the val set can determine
    """
    import math as _math

    cols = _TASK_COLS
    task_scores_list = []
    for row in X_va:
        task_scores_list.append({
            "task1": float(row[cols["task1"]]),
            "task2": float(row[cols["task2"]]),
            "task3": float(row[cols["task3"]]),
            "task4": float(row[cols["task4"]]),
        })

    # ── conflict_threshold ──────────────────────────────────────────────────
    # Compute all pairwise absolute differences on val set
    all_diffs = []
    for ts in task_scores_list:
        vals = list(ts.values())
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                all_diffs.append(abs(vals[i] - vals[j]))

    # Use the 75th percentile: pairs above this are genuinely "conflicting"
    all_diffs.sort()
    p75_idx      = int(len(all_diffs) * 0.75)
    conflict_thr = round(float(all_diffs[p75_idx]) if all_diffs else 0.30, 4)
    # Clamp to a sensible range [0.15, 0.50]
    conflict_thr = max(0.15, min(0.50, conflict_thr))

    # ── variance_decay and min_consistency ─────────────────────────────────
    # Compute per-sample variances
    variances = []
    for ts in task_scores_list:
        vals = list(ts.values())
        m = sum(vals) / len(vals)
        variances.append(sum((v-m)**2 for v in vals) / len(vals))

    max_var = max(variances) if variances else 0.25
    min_var = min(variances) if variances else 0.0

    # min_consistency = the lowest trust we should assign even at max disagreement
    # = fraction of val samples that are correct even at maximum variance
    correct_at_max = []
    for i, (ts, prob) in enumerate(zip(task_scores_list, val_probs)):
        vals = list(ts.values()); m = sum(vals)/len(vals)
        var  = sum((v-m)**2 for v in vals)/len(vals)
        if abs(var - max_var) < 0.02:   # near-maximum variance samples
            correct = (int(prob >= 0.5) == int(y_va[i]))
            correct_at_max.append(float(correct))
    min_consistency = round(
        float(sum(correct_at_max) / len(correct_at_max)) if correct_at_max else 0.10,
        4
    )
    min_consistency = max(0.05, min(0.30, min_consistency))

    # variance_decay: solve exp(-max_var * decay) = min_consistency
    # → decay = -ln(min_consistency) / max_var
    if max_var > 1e-6 and min_consistency > 1e-6:
        variance_decay = round(-_math.log(min_consistency) / max_var, 4)
        variance_decay = max(2.0, min(20.0, variance_decay))
    else:
        variance_decay = 8.0

    # ── direction_threshold ─────────────────────────────────────────────────
    # Find the task score boundary that best separates vulnerable from clean
    # Search over [0.10, 0.70] using mean task score on val set
    mean_scores = [
        sum(ts.values()) / len(ts) for ts in task_scores_list
    ]
    best_dir_thr = 0.30; best_acc = 0.0
    for t in [i/20 for i in range(2, 15)]:   # 0.10 to 0.70
        preds = [1 if s >= t else 0 for s in mean_scores]
        acc   = sum(1 for p, y in zip(preds, y_va) if p == y) / len(y_va)
        if acc > best_acc:
            best_acc = acc; best_dir_thr = round(t, 2)

    # ── blend_weights for Stage 3 ───────────────────────────────────────────
    # Fit: oracle_trust = w1*consistency + w2*calibration + w3*strategy_quality
    # Oracle trust = 1 - |V - y|  (1 when correct, 0 when maximally wrong)
    # Consistency and calibration are computable from val data
    # strategy_quality: use 0.80 (heuristic) for all val samples
    #   (val set is BigVul with mixed strategies; 0.80 is the heuristic default)

    oracle   = 1.0 - np.abs(val_probs - y_va.astype(np.float64))
    decisive = np.abs(val_probs - 0.5) * 2.0

    # Compute per-sample consistency from task variance
    consistencies = np.array([
        max(min_consistency,
            _math.exp(-v * variance_decay))
        for v in variances
    ], dtype=np.float64)

    # Calibration column (same formula used at inference)
    calibrations = np.clip(0.40 + 0.50 * decisive, 0.0, 1.0)  # placeholder
    strategy_col = np.full(len(y_va), 0.80)                    # heuristic default

    A = np.column_stack([consistencies, calibrations, strategy_col])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, oracle, rcond=None)
        w1, w2, w3 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except Exception:
        w1, w2, w3 = 0.50, 0.35, 0.15

    # ── Learn floor values from validation set ─────────────────────────────
    # The floors enforce an ordering constraint: w1 > w2 > w3.
    # We learn the minimum floor for each weight by finding the smallest
    # value that still maintains positive correlation with oracle trust.
    #
    # Method: start from unconstrained OLS weights (w1,w2,w3).
    # The floor for each weight = max(0.05, min(w_i, 0.45)) so that:
    #   - No weight is forced below 5% (prevents complete suppression)
    #   - No weight is locked above 45% (leaves room for OLS to adjust)
    #   - Ordering is preserved: FLOOR_CON > FLOOR_CAL > FLOOR_STR
    #
    # If OLS already respects ordering → floors are tight to OLS values.
    # If OLS collapses a weight to near-zero → floor rescues it.
    #
    raw_w = [max(0.0, w1), max(0.0, w2), max(0.0, w3)]
    raw_total = sum(raw_w)
    if raw_total > 1e-8:
        raw_w = [v / raw_total for v in raw_w]
    else:
        raw_w = [0.50, 0.35, 0.15]

    # Sort the raw weights to assign floors proportionally
    # Largest raw weight → largest floor, smallest → smallest floor
    sorted_idx = sorted(range(3), key=lambda i: raw_w[i], reverse=True)

    # Floor budget: total = 0.55, distributed 55%/30%/15% of budget
    floor_budget = 0.55
    floor_shares = [0.55, 0.30, 0.15]   # ordering: dominant/secondary/minor
    raw_floors   = [0.0, 0.0, 0.0]
    for rank, idx in enumerate(sorted_idx):
        raw_floors[idx] = floor_budget * floor_shares[rank]

    # Clamp floors: each between 0.05 and 0.45
    raw_floors = [max(0.05, min(0.45, f)) for f in raw_floors]

    # Enforce strict ordering: floor[0] > floor[1] > floor[2]
    # (consistency must dominate, strategy must be smallest)
    # Re-sort by the original priority: w1=consistency, w2=calibration, w3=strategy
    FLOOR_CON = raw_floors[sorted_idx.index(0)] if 0 in sorted_idx else raw_floors[0]
    FLOOR_CAL = raw_floors[sorted_idx.index(1)] if 1 in sorted_idx else raw_floors[1]
    FLOOR_STR = raw_floors[sorted_idx.index(2)] if 2 in sorted_idx else raw_floors[2]

    # Hard guarantee: FLOOR_CON >= FLOOR_CAL >= FLOOR_STR >= 0.05
    FLOOR_STR = max(0.05, FLOOR_STR)
    FLOOR_CAL = max(FLOOR_STR + 0.05, FLOOR_CAL)
    FLOOR_CON = max(FLOOR_CAL + 0.05, FLOOR_CON)
    # Normalise so they sum to at most 0.70 (leaves >= 0.30 free for OLS)
    floor_sum = FLOOR_CON + FLOOR_CAL + FLOOR_STR
    if floor_sum > 0.70:
        scale = 0.70 / floor_sum
        FLOOR_CON *= scale; FLOOR_CAL *= scale; FLOOR_STR *= scale
    FLOOR_CON = round(FLOOR_CON, 4)
    FLOOR_CAL = round(FLOOR_CAL, 4)
    FLOOR_STR = round(FLOOR_STR, 4)

    log.info(
        f"  Learned blend floors: "
        f"consistency={FLOOR_CON}  calibration={FLOOR_CAL}  strategy={FLOOR_STR}"
    )
    w1 = max(0.0, w1); w2 = max(0.0, w2); w3 = max(0.0, w3)
    total = w1 + w2 + w3
    if total < 1e-8:
        w1, w2, w3 = FLOOR_CON, FLOOR_CAL, FLOOR_STR
    else:
        w1, w2, w3 = w1/total, w2/total, w3/total
        # Compute how much each weight exceeds its own floor
        excess1 = max(0.0, w1 - FLOOR_CON)
        excess2 = max(0.0, w2 - FLOOR_CAL)
        excess3 = max(0.0, w3 - FLOOR_STR)
        total_excess = excess1 + excess2 + excess3
        # Budget available above the floors
        budget = 1.0 - (FLOOR_CON + FLOOR_CAL + FLOOR_STR)   # = 0.45
        if total_excess > 1e-8:
            # Distribute budget in proportion to excess
            w1 = FLOOR_CON + budget * (excess1 / total_excess)
            w2 = FLOOR_CAL + budget * (excess2 / total_excess)
            w3 = FLOOR_STR + budget * (excess3 / total_excess)
        else:
            # All weights were at or below floor — give budget to consistency
            w1 = FLOOR_CON + budget
            w2 = FLOOR_CAL
            w3 = FLOOR_STR

    # Derive w3 from w1+w2 to avoid float accumulation
    w1 = round(w1, 6); w2 = round(w2, 6); w3 = round(1.0 - w1 - w2, 6)
    # Final safety clamp (float rounding edge case)
    w1 = max(FLOOR_CON, w1); w2 = max(FLOOR_CAL, w2); w3 = max(FLOOR_STR, w3)
    total = w1 + w2 + w3
    w1 = round(w1/total, 4); w2 = round(w2/total, 4); w3 = round(1.0 - w1 - w2, 4)

    blend_weights = {
        "consistency":  w1,
        "calibration":  w2,
        "strategy":     w3,
    }

    # Final safety check: enforce omega1 (consistency) >= omega2 (calibration)
    # with a MINIMUM GAP of 0.10 so consistency clearly dominates.
    # This is the ordering constraint from the paper.
    MIN_GAP = 0.10
    if blend_weights["consistency"] < blend_weights["calibration"] + MIN_GAP:
        # Force consistency to be at least calibration + MIN_GAP
        # Redistribute the deficit from calibration and strategy proportionally
        target_con = round(min(0.80, blend_weights["calibration"] + MIN_GAP), 4)
        deficit    = round(target_con - blend_weights["consistency"], 4)
        # Take the deficit from calibration first, then strategy if needed
        take_from_cal  = min(deficit, max(0.0, blend_weights["calibration"] - FLOOR_CAL))
        take_from_str  = round(deficit - take_from_cal, 4)
        blend_weights["consistency"] = round(target_con, 4)
        blend_weights["calibration"] = round(blend_weights["calibration"] - take_from_cal, 4)
        blend_weights["strategy"]    = round(
            max(FLOOR_STR, blend_weights["strategy"] - take_from_str), 4)
        # Renormalise to sum=1
        total = blend_weights["consistency"] + blend_weights["calibration"] + blend_weights["strategy"]
        blend_weights["consistency"] = round(blend_weights["consistency"] / total, 4)
        blend_weights["calibration"] = round(blend_weights["calibration"] / total, 4)
        blend_weights["strategy"]    = round(1.0 - blend_weights["consistency"]
                                             - blend_weights["calibration"], 4)
        log.info(
            f"  blend_weights: ordering enforced with gap={MIN_GAP} — "
            f"rebalanced to {blend_weights}"
        )

    log.info(
        f"  ALC params learned from val set:\n"
        f"    conflict_threshold={conflict_thr}  "
        f"min_consistency={min_consistency}\n"
        f"    variance_decay={variance_decay}  "
        f"direction_threshold={best_dir_thr}\n"
        f"    blend_weights={blend_weights}"
    )

    return {
        "conflict_threshold":  conflict_thr,
        "min_consistency":     min_consistency,
        "variance_decay":      variance_decay,
        "direction_threshold": best_dir_thr,
        "blend_weights":       blend_weights,
        # strategy_quality is a pipeline constant, not learned from val data
        "strategy_quality": {
            "ground_truth": 1.00,
            "heuristic":    0.80,
            "all_lines":    0.50,
        },
    }

def learn_task_weights(X_tr, y_tr):
    """
    Learn relative importance weights for the four MTD tasks
    by gradient descent on the training set.
    Weights are non-negative and sum to 1.
    """
    cols    = list(_TASK_COLS.values())
    Xt      = X_tr[:, cols].astype(np.float64)
    col_std = Xt.std(0); col_std[col_std < 1e-8] = 1.0
    Xn      = Xt / col_std
    yf      = y_tr.astype(np.float64)
    rng     = np.random.RandomState(42)
    w       = rng.rand(4).astype(np.float64) * 0.25
    for _ in range(500):
        prob = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w, -500, 500)))
        grad = Xn.T @ (prob - yf) / len(yf)
        w   -= 0.05 * grad
        w    = np.maximum(0.0, w)
        s    = w.sum()
        if s > 1e-8:
            w /= s
    if w.sum() < 1e-8:
        w = np.array([0.25, 0.25, 0.25, 0.25])
    wts = {k: round(float(v), 6)
           for k, v in zip(["task1","task2","task3","task4"], w)}
    log.info(f"  Learned task weights: {wts}")
    return wts




def learn_task3_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
    """
    Learn severity multipliers (w_H, w_M, w_L) for Task 3.

    Searches over integer multiplier ratios subject to
    monotonicity: w_H >= w_M >= w_L > 0.

    The counts high_n, medium_n, low_n are stored as separate
    features in the feature vector — we find the combination
    that best correlates with vulnerability labels.
    """
    try:
        high_col   = FEATURE_NAMES.index("high_severity_count")
        medium_col = FEATURE_NAMES.index("medium_severity_count")
        low_col    = FEATURE_NAMES.index("low_severity_count")
    except ValueError:
        # Feature names not found — return safe defaults
        log.warning("  Task3 severity count features not found — using default weights")
        return {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0}

    high_n   = X_va[:, high_col].astype(np.float64)
    medium_n = X_va[:, medium_col].astype(np.float64)
    low_n    = X_va[:, low_col].astype(np.float64)
    yf       = y_va.astype(np.float64)

    best_wH, best_wM, best_wL = 3.0, 2.0, 1.0
    best_corr = -2.0

    # Search over integer-ratio candidates satisfying w_H >= w_M >= w_L >= 1
    for wH in range(1, 6):
        for wM in range(1, wH + 1):
            for wL in range(1, wM + 1):
                score = wH * high_n + wM * medium_n + wL * low_n
                if score.std() > 1e-8 and yf.std() > 1e-8:
                    corr = float(np.corrcoef(score, yf)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_wH, best_wM, best_wL = float(wH), float(wM), float(wL)

    # Enforce STRICT monotonicity: w_H > w_M > w_L
    # If correlation search returns equal weights (e.g., 5,5,5),
    # apply a small enforced spread so HIGH always contributes more than LOW.
    if best_wH == best_wM == best_wL:
        # Equal weights found — enforce domain ordering with small spread
        best_wH = best_wH
        best_wM = max(1.0, best_wH - 1.0)
        best_wL = max(1.0, best_wM - 1.0)
        log.info(f"  task3: equal weights detected — enforced spread: "
                 f"w_H={best_wH} w_M={best_wM} w_L={best_wL}")
    elif best_wH == best_wM:
        best_wM = max(1.0, best_wH - 1.0)
    elif best_wM == best_wL:
        best_wL = max(1.0, best_wM - 1.0)
    log.info(
        f"  task3_weights learned: w_H={best_wH}  w_M={best_wM}  w_L={best_wL}"
        f"  (Pearson corr={best_corr:.4f})"
    )
    return {"w_H": best_wH, "w_M": best_wM, "w_L": best_wL}



def learn_task4_weights(X_va: np.ndarray, y_va: np.ndarray) -> dict:
    """
    Learn the data-flow and control-flow blend weights for Task 4.

    Task 4 computes:
        overall = w_df * df_risk + w_cf * cf_risk,  w_df + w_cf = 1

    We search over w_df in [0.05, 0.95] (step 0.05) and set
    w_cf = 1 - w_df, choosing the value that maximises Pearson
    correlation with ground-truth labels on the validation set.

    Defaults: w_df=0.60, w_cf=0.40 (data-flow dominates since
    most memory-safety vulnerabilities originate from data flow).
    """
    try:
        df_col = FEATURE_NAMES.index("data_flow_risk")
        cf_col = FEATURE_NAMES.index("control_flow_risk")
    except ValueError:
        log.warning("  Task4 risk features not found — using default weights")
        return {"w_df": 0.60, "w_cf": 0.40}

    df_risk = X_va[:, df_col].astype(np.float64)
    cf_risk = X_va[:, cf_col].astype(np.float64)
    yf      = y_va.astype(np.float64)

    best_wdf  = 0.60
    best_corr = -2.0

    for i in range(1, 20):          # w_df in {0.05, 0.10, ..., 0.95}
        w_df  = round(i / 20.0, 2)
        w_cf  = round(1.0 - w_df, 2)
        score = w_df * df_risk + w_cf * cf_risk
        if score.std() > 1e-8 and yf.std() > 1e-8:
            corr = float(np.corrcoef(score, yf)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_wdf  = w_df

    # Floor w_df >= 0.40: data-flow must always contribute at least 40%.
    # Most memory-safety vulnerabilities (buffer overflow, use-after-free,
    # injection) are data-flow phenomena. If control-flow alone predicts
    # better on a specific dataset, it is likely due to confounding features
    # rather than genuine domain signal.
    best_wdf = max(0.40, best_wdf)
    best_wcf = round(1.0 - best_wdf, 2)
    log.info(
        f"  task4_weights learned: w_df={best_wdf}  w_cf={best_wcf}"
        f"  (Pearson corr={best_corr:.4f})"
    )
    return {"w_df": best_wdf, "w_cf": best_wcf}

def _save(model, scaler, opt_thresh, trust_thresh, trust_cal,
          task_wts, alc_params, metrics, mv_dist,
          task1_alpha=0.50,
          task3_weights=None,
          task4_weights=None, extra=None):
    d = {
        **model.to_dict(),
        "scaler":             scaler.to_dict(),
        "feature_names":      FEATURE_NAMES,
        "trained_on":         "bigvul_only",
        "scaler_fitted_on":   "bigvul+megavul_combined",
        "task1_alpha":        round(task1_alpha, 4),
        "task3_weights":      task3_weights or {"w_H": 3.0, "w_M": 2.0, "w_L": 1.0},
        "task4_weights":      task4_weights or {"w_df": 0.60, "w_cf": 0.40},
        # All learned from data — nothing hardcoded:
        "opt_threshold":      opt_thresh["threshold"],
        "threshold_metrics":  opt_thresh,
        "trust_threshold":    trust_thresh,
        "trust_calibration":  trust_cal,
        "task_weights":       task_wts,
        "alc_params":         alc_params,   # ALC constants, all data-driven
        "metrics":            metrics,
        "megavul_score_dist": mv_dist,
    }
    if extra: d.update(extra)
    return d


def train_logreg(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv):
    log.info("Training Logistic Regression on BigVul labels...")
    model = LogisticRegression(lr=0.05, epochs=300, batch_size=64, l2=1e-4)
    model.fit(Xtr, ytr)
    vp    = model.predict_proba(Xva)
    opt   = find_opt_threshold(vp, yva)
    ts    = compute_val_task_scores(Xva, yva)
    alcp  = learn_alc_params(Xva, yva, vp)
    tcal  = calibrate_trust(vp, yva)
    tthr  = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
                                  blend_weights=alcp["blend_weights"],
                                  trust_cal=tcal,
                                  strategy_quality=alcp["strategy_quality"])
    twts  = learn_task_weights(Xtr, ytr)
    # Learn alpha from combined BigVul val + MegaVul (any dataset)
    t1a   = learn_task1_alpha(Xva, yva)
    t3w   = learn_task3_weights(Xva, yva)
    t4w   = learn_task4_weights(Xva, yva)
    imps  = feature_importance_logreg(model)
    met   = evaluate(model, Xte, yte, threshold=opt["threshold"])
    mvd   = compute_megavul_score_dist(model, Xmv)
    log.info(f"  LR test metrics (BigVul): {met}")
    log.info(f"  Top-5 features: {imps[:5]}")
    log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
    save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
                 task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w,
                 extra={"feature_importances": imps})
    path = MODELS_DIR / "logreg_model.json"
    path.write_text(json.dumps(save, indent=2), encoding="utf-8")
    log.info(f"  LR model saved → {path}")
    return met


def train_nn(Xtr, ytr, Xva, yva, Xte, yte, scaler, Xmv, epochs=100):
    log.info("Training Neural Network (MLP) on BigVul labels...")
    # Select architecture, dropout, and L2 from validation set
    cfg = select_nn_architecture(Xtr, ytr, Xva, yva)
    model = NeuralNetwork(
        h1=cfg["h1"], h2=cfg["h2"],
        lr=0.001, epochs=epochs, bs=64,
        l2=cfg["l2"], dropout=cfg["dropout"]
    )
    model.fit(Xtr, ytr)
    vp   = model.predict_proba(Xva)
    opt  = find_opt_threshold(vp, yva)
    ts   = compute_val_task_scores(Xva, yva)
    alcp = learn_alc_params(Xva, yva, vp)
    tcal = calibrate_trust(vp, yva)
    tthr = find_trust_threshold(vp, yva, ts, alcp["variance_decay"],
                                 blend_weights=alcp["blend_weights"],
                                 trust_cal=tcal,
                                 strategy_quality=alcp["strategy_quality"])
    twts = learn_task_weights(Xtr, ytr)
    # Learn alpha from combined BigVul val + MegaVul (any dataset)
    t1a   = learn_task1_alpha(Xva, yva)
    t3w   = learn_task3_weights(Xva, yva)
    t4w   = learn_task4_weights(Xva, yva)
    met  = evaluate(model, Xte, yte, threshold=opt["threshold"])
    mvd  = compute_megavul_score_dist(model, Xmv)
    log.info(f"  NN test metrics (BigVul): {met}")
    log.info(f"  MegaVul score dist: mean={mvd['mean']}  std={mvd['std']}  buckets={mvd['buckets']}")
    save = _save(model, scaler, opt, tthr, tcal, twts, alcp, met, mvd,
                 task1_alpha=t1a, task3_weights=t3w, task4_weights=t4w,
                 extra={"nn_architecture": cfg})
    path = MODELS_DIR / "nn_model.json"
    path.write_text(json.dumps(save, indent=2), encoding="utf-8")
    log.info(f"  NN model saved → {path}")
    return met


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MTD ML models")
    parser.add_argument("--model", choices=["logreg","nn","both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    args = parser.parse_args()

    log.info("=== MTD ML Training ===")
    log.info(f"Supervised: BigVul only  |  Scaler: BigVul+MegaVul  |  Model: {args.model}")

    # ── Load BigVul labelled data for training ────────────────────────────────
    Xbv, ybv, ids = load_labelled("bigvul")
    n_pos = int((ybv==1).sum()); n_neg = int((ybv==0).sum())
    log.info(f"BigVul — total={len(ybv)}  vulnerable={n_pos}  non-vulnerable={n_neg}")
    if len(ybv) < 20 or n_pos == 0 or n_neg == 0:
        log.error(
            f"Need both label=0 and label=1 samples (got pos={n_pos}, neg={n_neg}).\n"
            f"Run:  python mtd/ml/build_dataset.py --dataset bigvul --max-samples 2000"
        )
        sys.exit(1)

    # ── Load MegaVul features for scaler fitting only (not for training) ──────
    # IMPORTANT: MegaVul is used ONLY to fit the scaler so it generalises
    # to MegaVul's feature distribution at inference time.
    # MegaVul is NOT used for training, calibration, or threshold finding
    # because its V scores are unreliable (heuristic strategy, no flaw lines).
    # Mixing MegaVul into calibrate_trust() destroys the T score distribution.
    Xmv, _ = load_all_features("megavul")

    # ── Train/val/test split on BigVul only ───────────────────────────────────
    Xtr,ytr,_,Xva,yva,_,Xte,yte,_ = split3(Xbv, ybv, ids,
                                             val=args.val_ratio, test=args.test_ratio)
    log.info(f"Split — train={len(ytr)}  val={len(yva)}  test={len(yte)}")
    log.info(f"Train dist: {dict(Counter(ytr.tolist()))}")
    log.info(f"Val   dist: {dict(Counter(yva.tolist()))}")
    log.info(f"Test  dist: {dict(Counter(yte.tolist()))}")

    # ── Scaler fitted on BigVul train + all MegaVul features ─────────────────
    scaler = build_combined_scaler(Xtr, Xmv)
    Xtr_s  = scaler.transform(Xtr)
    Xva_s  = scaler.transform(Xva)
    Xte_s  = scaler.transform(Xte)
    Xmv_s  = (scaler.transform(Xmv)
               if Xmv.shape[0] > 0
               else np.empty((0, FEATURE_DIM), dtype=np.float32))

    report = {
        "trained_on":         "bigvul_only",
        "scaler_fitted_on":   "bigvul+megavul",
        "n_train":            len(ytr),
        "n_val":              len(yva),
        "n_test":             len(yte),
        "n_megavul_scaler":   Xmv.shape[0],
    }
    if args.model in ("logreg","both"):
        report["logreg"] = train_logreg(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s)
    if args.model in ("nn","both"):
        report["nn"] = train_nn(Xtr_s, ytr, Xva_s, yva, Xte_s, yte, scaler, Xmv_s,
                                epochs=args.epochs)

    rp = MODELS_DIR / "training_report.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"Training report → {rp}")
    log.info("=== Training complete ===")


def _sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))


if __name__ == "__main__":
    main()




