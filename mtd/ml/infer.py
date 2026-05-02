# =============================================================================
# mtd/ml/infer.py  —  ML Inference Module
# All scoring parameters loaded from trained model files — nothing hardcoded.
# =============================================================================

import json, logging, math
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parent / "models"


class ModelNotTrainedError(RuntimeError):
    pass


class MLScorer:
    def __init__(self, path):
        self._path=Path(path); self._data=None; self._loaded=False
        self._sm=None; self._ss=None
    def _load(self):
        if self._loaded: return
        if not self._path.exists():
            raise ModelNotTrainedError(
                f"Model not found: {self._path}\n"
                f"Run training first:\n"
                f"  python mtd/ml/build_dataset.py --dataset all\n"
                f"  python mtd/ml/train.py --model both"
            )
        try: self._data=json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e: raise ModelNotTrainedError(f"Cannot read {self._path}: {e}")
        sc=self._data.get("scaler",{})
        if not sc.get("mean"): raise ModelNotTrainedError(f"Missing scaler in {self._path}. Re-run training.")
        self._sm=np.array(sc["mean"],dtype=np.float32); self._ss=np.array(sc["std"],dtype=np.float32)
        self._loaded=True
        log.info(f"Model loaded: {self._data['model_type']} from {self._path}")
    def is_ready(self):
        try: self._load(); return True
        except ModelNotTrainedError: return False
    def score(self, features: list) -> float:
        self._load()
        x=np.array(features,dtype=np.float32)
        ss=self._ss.copy(); ss[ss<1e-8]=1.0; x=(x-self._sm)/ss
        mt=self._data["model_type"]
        if mt=="logistic_regression":
            w=np.array(self._data["weights"],dtype=np.float32); b=float(self._data["bias"])
            return float(_sig(float(x@w+b)))
        elif mt=="neural_network":
            p={k:np.array(v,dtype=np.float32) for k,v in self._data["params"].items()}
            a1=np.maximum(0,x@p["W1"]+p["b1"]); a2=np.maximum(0,a1@p["W2"]+p["b2"])
            return float(_sig(float(a2@p["W3"].flatten()+p["b3"].flatten()[0])))
        raise ModelNotTrainedError(f"Unknown model type: {mt}")
    def _get(self, key):
        self._load()
        if key not in self._data:
            raise ModelNotTrainedError(f"'{key}' missing in {self._path}. Re-run training.")
        return self._data[key]
    def get_opt_threshold(self):    return float(self._get("opt_threshold"))
    def get_trust_threshold(self):  return float(self._get("trust_threshold"))
    def get_trust_calibration(self):return self._get("trust_calibration")
    def get_task_weights(self):     return self._get("task_weights")
    def get_alc_params(self):      return self._get("alc_params")
    def get_feature_importances(self): return self._data.get("feature_importances",[]) if self._loaded else []
    def get_metrics(self): return self._data.get("metrics",{}) if self._loaded else {}


_lr = MLScorer(MODELS_DIR/"logreg_model.json")
_nn = MLScorer(MODELS_DIR/"nn_model.json")


def _require():
    if not _lr.is_ready() and not _nn.is_ready():
        raise ModelNotTrainedError(
            "No trained models found.\n"
            "Run:\n  python mtd/ml/build_dataset.py --dataset all\n"
            "     python mtd/ml/train.py --model both"
        )


def score_ensemble(features: list) -> float:
    _require()
    s = [m.score(features) for m in [_lr,_nn] if m.is_ready()]
    return round(sum(s)/len(s), 6)


def _avg(fn):
    _require()
    vals = [getattr(m,fn)() for m in [_lr,_nn] if m.is_ready()]
    return vals


def get_opt_threshold() -> float:
    return round(sum(_avg("get_opt_threshold"))/len(_avg("get_opt_threshold")), 4)


def get_trust_threshold() -> float:
    """The learned threshold below which T triggers 'untrustworthy'."""
    vals = _avg("get_trust_threshold")
    return round(sum(vals)/len(vals), 4)


def get_trust_calibration() -> dict:
    vals = _avg("get_trust_calibration")
    return {"intercept": round(sum(v["intercept"] for v in vals)/len(vals),6),
            "slope":     round(sum(v["slope"]     for v in vals)/len(vals),6)}


def get_alc_params() -> dict:
    """Return all ALC constants learned from the validation set."""
    vals = _avg("get_alc_params")
    if len(vals) == 1:
        return vals[0]
    # Average numeric leaf values across LR and NN
    result = {}
    for key in vals[0]:
        v0, v1 = vals[0][key], vals[1][key]
        if isinstance(v0, dict):
            result[key] = {k: round((v0[k]+v1[k])/2, 6) for k in v0}
        elif isinstance(v0, (int, float)):
            result[key] = round((float(v0)+float(v1))/2, 6)
        else:
            result[key] = v0   # take first for non-numeric (strategy_quality dict)
    return result


def get_task_weights() -> dict:
    vals = _avg("get_task_weights"); keys=["task1","task2","task3","task4"]
    avg  = {k: round(sum(v[k] for v in vals)/len(vals),6) for k in keys}
    tot  = sum(avg.values())
    if tot>1e-8: avg={k:round(v/tot,6) for k,v in avg.items()}
    return avg


def models_available() -> dict:
    return {"logreg":_lr.is_ready(),"nn":_nn.is_ready()}


def _sig(x): return 1.0/(1.0+math.exp(-max(-500.0,min(500.0,x))))

