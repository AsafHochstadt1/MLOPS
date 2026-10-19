"""
Mini MLOps demo: the full loop in one file, no external services.

Stages shown:
  1. Data versioning        (hash of the dataset)
  2. Experiment tracking    (JSON-lines log of params + metrics)
  3. Training pipeline      (train -> evaluate)
  4. Validation gate        (only promote if better than current production)
  5. Model registry         (versioned models + a "production" pointer)
  6. Serving                (load production model and predict)
  7. Monitoring / drift     (PSI + accuracy check on live data)
  8. Automated retraining   (triggered when drift is detected)

Requires: numpy, scikit-learn, joblib   (pip install scikit-learn)
Run:      python mlops_demo.py
"""
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path("mlops_workspace")
REGISTRY = ROOT / "registry"
TRACKING_LOG = ROOT / "experiments.jsonl"
PROD_POINTER = REGISTRY / "production.json"
ROOT.mkdir(exist_ok=True)
REGISTRY.mkdir(exist_ok=True)


# ---------- Simulated data source ----------
def make_data(n=2000, shift=0.0, seed=0):
    """Two features; the label depends on both. `shift` simulates the
    real world changing (feature drift + changed relationship)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=[0, 0], scale=1, size=(n, 2))
    X[:, 0] += shift                      # feature distribution moves
    y = (X[:, 0] - shift * 0.5 + X[:, 1] > 0).astype(int)
    flip = rng.random(n) < 0.05           # 5% label noise
    y[flip] = 1 - y[flip]
    return X, y


# ---------- 1. Data versioning ----------
def data_version(X, y):
    return hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:10]


# ---------- 2. Experiment tracking ----------
def log_experiment(record):
    with open(TRACKING_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------- 5. Model registry ----------
def next_version():
    return len(list(REGISTRY.glob("model_v*.joblib"))) + 1


def register(model, meta):
    v = next_version()
    joblib.dump(model, REGISTRY / f"model_v{v}.joblib")
    (REGISTRY / f"model_v{v}.json").write_text(json.dumps(meta, indent=2))
    return v


def promote(version):
    PROD_POINTER.write_text(json.dumps({"version": version}))
    print(f"   -> promoted model v{version} to PRODUCTION")


def production_info():
    if not PROD_POINTER.exists():
        return None
    v = json.loads(PROD_POINTER.read_text())["version"]
    meta = json.loads((REGISTRY / f"model_v{v}.json").read_text())
    return v, meta


# ---------- 3 + 4. Training pipeline with validation gate ----------
def training_pipeline(X, y, reason):
    print(f"\n[PIPELINE] Training triggered ({reason})")
    dv = data_version(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)

    candidates = {
        "logreg": LogisticRegression(),
        "random_forest": RandomForestClassifier(n_estimators=50, random_state=1),
    }
    best_name, best_model, best_acc = None, None, -1
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        log_experiment({"time": time.time(), "model": name,
                        "data_version": dv, "accuracy": round(acc, 4)})
        print(f"   experiment: {name:14s} accuracy={acc:.3f}  data={dv}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    # Validation gate: compare against current production on the SAME new test set
    current = production_info()
    if current:
        prod_model = joblib.load(REGISTRY / f"model_v{current[0]}.joblib")
        prod_acc = accuracy_score(y_te, prod_model.predict(X_te))
        print(f"   gate: candidate={best_acc:.3f} vs production v{current[0]}={prod_acc:.3f}")
        if best_acc <= prod_acc:
            print("   -> candidate NOT better; keeping current production model")
            return
    meta = {"algorithm": best_name, "accuracy": round(best_acc, 4),
            "data_version": dv, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_feature_means": X_tr.mean(axis=0).tolist(),
            "train_feature_stds": X_tr.std(axis=0).tolist(),
            "reference_sample": X_tr[:500, 0].tolist()}
    v = register(best_model, meta)
    promote(v)


# ---------- 6. Serving ----------
def predict(X):
    v, _ = production_info()
    model = joblib.load(REGISTRY / f"model_v{v}.joblib")
    return model.predict(X)


# ---------- 7. Monitoring ----------
def psi(reference, current, bins=10):
    """Population Stability Index: common drift metric.
    <0.1 stable, 0.1-0.25 moderate drift, >0.25 major drift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monitor(X_live, y_live, label):
    v, meta = production_info()
    score = psi(np.array(meta["reference_sample"]), X_live[:, 0])
    acc = accuracy_score(y_live, predict(X_live))
    status = "OK" if score < 0.25 and acc > meta["accuracy"] - 0.05 else "ALERT"
    print(f"\n[MONITOR] {label}: model v{v} | PSI={score:.3f} | "
          f"live accuracy={acc:.3f} (baseline {meta['accuracy']:.3f}) -> {status}")
    return status == "ALERT"


# ---------- Orchestration: the whole lifecycle ----------
if __name__ == "__main__":
    # Day 1: initial training on historical data
    X0, y0 = make_data(seed=0)
    training_pipeline(X0, y0, reason="initial training")

    # Week 2: live traffic looks like training data -> healthy
    X1, y1 = make_data(n=800, seed=10)
    monitor(X1, y1, "week 2 (normal traffic)")

    # Week 10: the world changes (drift)
    X2, y2 = make_data(n=800, shift=2.0, seed=20)
    drifted = monitor(X2, y2, "week 10 (world has changed)")

    # Automated response: retrain on fresh data, gate decides whether to promote
    if drifted:
        Xn, yn = make_data(n=2000, shift=2.0, seed=30)
        training_pipeline(Xn, yn, reason="drift alert")
        monitor(X2, y2, "week 10 after retraining")

    print(f"\nArtifacts written to ./{ROOT}/  (registry + experiments.jsonl)")