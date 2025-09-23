# ============================================================
# Lab07 A1–A5 — Classification on Sleep-EDF Dataset
# Modularized code with fixes for consistent outputs
# ============================================================

import os
import numpy as np
import pandas as pd
import mne
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ---------------------------
# USER SETTINGS
# ---------------------------
DATA_PATH = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0"
PREFERRED_CHANNELS = ["EEG Fpz-Cz", "Fpz-Cz", "EEG Pz-Oz", "Pz-Oz"]
CROP_SECONDS = None             # None = full night, else crop in seconds
MAX_EPOCHS_PER_FILE = 500       # limit epochs per subject
EPOCH_SEC = 30                  # 30-second windows

# ---------------------------
# Utilities
# ---------------------------
def load_raw(psg_file, hyp_file=None, verbose=True):
    """Load PSG EDF file + hypnogram annotations."""
    if verbose:
        print("Loading PSG:", psg_file)
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose="ERROR")
    if CROP_SECONDS:
        raw.crop(tmin=0, tmax=CROP_SECONDS)
    if hyp_file and os.path.exists(hyp_file):
        ann = mne.read_annotations(hyp_file)
        raw.set_annotations(ann)
    return raw

def pick_best_channel(raw, preferred=PREFERRED_CHANNELS):
    """Pick EEG channel (prefer Fpz-Cz or Pz-Oz)."""
    for p in preferred:
        if p in raw.ch_names:
            raw.pick_channels([p])
            return p
    # fallback
    for ch in raw.ch_names:
        if "EEG" in ch:
            raw.pick_channels([ch])
            return ch
    raw.pick_channels([raw.ch_names[0]])
    return raw.ch_names[0]

def extract_epoch_features(raw, keep_stages=(2, 4), epoch_sec=EPOCH_SEC, max_epochs=MAX_EPOCHS_PER_FILE):
    """Extract epochs and compute statistical features."""
    mapping = {
        "Sleep stage W": 0, "W": 0,
        "Sleep stage 1": 1, "1": 1,
        "Sleep stage 2": 2, "2": 2, "N2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "3": 3, "4": 3,
        "Sleep stage R": 4, "R": 4, "REM": 4, "REM sleep": 4
    }

    sfreq = int(raw.info["sfreq"])
    sig = raw.get_data()[0]
    features, labels = [], []

    for i, ann in enumerate(raw.annotations):
        if max_epochs and len(features) >= max_epochs:
            break
        desc = ann["description"]
        lab = mapping.get(desc, -1)
        if lab == -1:  # ignore unknown
            continue
        start = int(ann["onset"] * sfreq)
        end = start + epoch_sec * sfreq
        if end > len(sig):
            continue
        seg = sig[start:end]
        feats = [
            np.mean(seg), np.std(seg),
            np.min(seg), np.max(seg),
            np.percentile(seg, 25), np.percentile(seg, 75)
        ]
        features.append(feats)
        labels.append(lab)

    if not features:
        return pd.DataFrame()  # empty

    df = pd.DataFrame(features, columns=["mean", "std", "min", "max", "p25", "p75"])
    df["stage"] = labels
    return df

# ---------------------------
# A1: Data Preparation
# ---------------------------
def run_A1(df):
    print("\n=== A1: Data Preparation ===")
    print("Shape:", df.shape)
    print("Class distribution:", Counter(df["stage"]))
    return df.drop(columns=["stage"]), df["stage"]

# ---------------------------
# A2: Hyperparameter Tuning
# ---------------------------
def run_A2(X_train, y_train):
    print("\n=== A2: Hyperparameter Tuning ===")
    models = {
        "DecisionTree": (DecisionTreeClassifier(), {"max_depth":[3,5,10,None]}),
        "RandomForest": (RandomForestClassifier(), {"n_estimators":[50,100], "max_depth":[3,5,None]}),
        "SVM": (SVC(), {"C":[0.1,1,10], "kernel":["linear","rbf"]}),
        "LogReg": (LogisticRegression(max_iter=1000), {"C":[0.1,1,10]})
    }
    tuned_models = {}
    for name, (model, param_grid) in models.items():
        search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=3, verbose=0)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        print(f"{name} best params:", search.best_params_)
    return tuned_models

# ---------------------------
# A3: Comparative Evaluation
# ---------------------------
def run_A3(models, X_train, y_train, X_test, y_test):
    print("\n=== A3: Comparative Evaluation ===")
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, digits=3, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        results.append((name, train_acc, test_acc, report, cm))
        print(f"\n{name}:")
        print("Train acc:", train_acc, " Test acc:", test_acc)
        print("Confusion matrix:\n", cm)
        print(classification_report(y_test, preds, digits=3))
    return results

# ---------------------------
# A4: Regression Placeholder
# ---------------------------
def run_A4():
    print("\n=== A4: Regression Task Placeholder ===")
    print("This section can include regression on continuous features (not implemented).")

# ---------------------------
# A5: Clustering Placeholder
# ---------------------------
def run_A5():
    print("\n=== A5: Clustering Task Placeholder ===")
    print("This section can include clustering (KMeans, Hierarchical, etc.) (not implemented).")

# ---------------------------
# Main pipeline
# ---------------------------
def main_run(psg_file, hyp_file):
    raw = load_raw(psg_file, hyp_file)
    ch = pick_best_channel(raw)
    print("Picked channel:", ch)
    df = extract_epoch_features(raw, keep_stages=(2,4))
    if df.empty:
        print("No N2/REM found, falling back to ALL stages.")
        df = extract_epoch_features(raw, keep_stages=(0,1,2,3,4))

    if df.empty:
        print("⚠️ Still no usable epochs, skipping file.")
        return

    # A1
    X, y = run_A1(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # A2
    tuned_models = run_A2(X_train, y_train)

    # A3
    results = run_A3(tuned_models, X_train, y_train, X_test, y_test)

    # A4
    run_A4()

    # A5
    run_A5()

def run_on_dataset(data_path, max_files=1):
    count = 0
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.endswith("-PSG.edf"):
                psg = os.path.join(root, f)
                hyp = None
                for g in files:
                    if "hyp" in g.lower():
                        hyp = os.path.join(root, g)
                        break
                print("\n=== Running file:", psg, "===")
                main_run(psg, hyp)
                count += 1
                if max_files and count >= max_files:
                    return
    print("\nLab07 run completed.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    run_on_dataset(DATA_PATH, max_files=1)
