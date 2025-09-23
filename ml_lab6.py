# Lab06 A1-A7 — Modularized functions using Sleep-EDF Expanded dataset
# Always prints outputs: falls back to all stages if N2/REM not found

import os
from collections import Counter
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# USER SETTINGS
# ---------------------------
DATA_PATH = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0"
PREFERRED_CHANNELS = ["EEG Fpz-Cz", "Fpz-Cz", "EEG Pz-Oz", "Pz-Oz"]
ENABLE_PLOTS = False        # Set True to see plots
CROP_SECONDS = None         # None = use full night, or set seconds (e.g. 7200)

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
    for ch in raw.ch_names:
        if "EEG" in ch or "Fpz" in ch or "Pz" in ch:
            raw.pick_channels([ch])
            return ch
    raw.pick_channels([raw.ch_names[0]])
    return raw.ch_names[0]

def extract_epoch_features(raw, keep_stages=(2, 4), epoch_sec=30, max_epochs=None):
    """Extract epochs and compute features. Falls back if N2/REM not found."""
    mapping = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
        "W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4
    }
    sfreq = int(raw.info["sfreq"])
    sig = raw.get_data()[0]
    features, labels = [], []

    for i, ann in enumerate(raw.annotations):
        if max_epochs and i >= max_epochs:
            break
        desc = ann["description"]
        lab = mapping.get(desc, -1)
        if lab == -1:
            continue
        if keep_stages and lab not in keep_stages:
            continue
        start = int(ann["onset"] * sfreq)
        end = start + epoch_sec * sfreq
        if end > len(sig):
            continue
        seg = sig[start:end]
        feats = [np.mean(seg), np.std(seg), np.min(seg),
                 np.max(seg), np.percentile(seg, 25), np.percentile(seg, 75)]
        features.append(feats)
        labels.append(lab)

    df = pd.DataFrame(features, columns=["mean", "std", "min", "max", "p25", "p75"])
    df["stage"] = labels
    return df

# ---------------------------
# A1: Entropy + Binning
# ---------------------------
def entropy(labels):
    counts = Counter(labels)
    n = sum(counts.values())
    return -sum((c/n)*log2(c/n) for c in counts.values() if c > 0)

def equal_width_binning(x, n_bins=4):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    codes = np.digitize(x, edges[1:-1])
    return codes, edges

def equal_freq_binning(x, n_bins=4):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    codes = np.digitize(x, edges[1:-1])
    return codes, edges

# ---------------------------
# A2: Gini Index
# ---------------------------
def gini_index(labels):
    counts = Counter(labels)
    n = sum(counts.values())
    return 1 - sum((c/n)**2 for c in counts.values())

# ---------------------------
# A3/A4: Information Gain + Root Attribute
# ---------------------------
def information_gain(y, x_cat):
    H_y = entropy(y)
    n = len(y)
    ig = H_y
    for v in np.unique(x_cat):
        mask = (x_cat == v)
        ig -= (np.sum(mask)/n) * entropy(np.array(y)[mask])
    return ig

def choose_root_attribute(X_df, y, binning="width", n_bins=4):
    ig_scores, binned = {}, pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        if binning == "freq":
            codes, _ = equal_freq_binning(X_df[col].values, n_bins)
        else:
            codes, _ = equal_width_binning(X_df[col].values, n_bins)
        binned[col] = codes
        ig_scores[col] = information_gain(y, codes)
    best = max(ig_scores, key=ig_scores.get)
    return best, ig_scores, binned

# ---------------------------
# A5: ID3 Tree
# ---------------------------
class SimpleDTNode:
    def __init__(self, depth=0):
        self.depth, self.is_leaf, self.pred = depth, False, None
        self.split_feature, self.children = None, {}

def build_id3_tree(X_binned, y, max_depth=5):
    def build(X, y, depth):
        node = SimpleDTNode(depth)
        if len(set(y)) == 1 or depth >= max_depth:
            node.is_leaf, node.pred = True, Counter(y).most_common(1)[0][0]
            return node
        best_feat = max(X.columns, key=lambda c: information_gain(y, X[c].values))
        node.split_feature = best_feat
        for val in np.unique(X[best_feat]):
            mask = X[best_feat] == val
            child = build(X.loc[mask].drop(columns=[best_feat]), np.array(y)[mask], depth+1)
            node.children[val] = child
        return node
    return build(X_binned, y, 0)

def predict_id3_batch(root, X_binned):
    preds = []
    for _, row in X_binned.iterrows():
        node = root
        while not node.is_leaf:
            val = row[node.split_feature]
            node = node.children.get(val, node)
        preds.append(node.pred if node.pred is not None else 0)
    return np.array(preds)

# ---------------------------
# A6: Sklearn Tree Visualization
# ---------------------------
def sklearn_tree_visualization(X_binned, y):
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_binned, y)
    if ENABLE_PLOTS:
        plt.figure(figsize=(12,6))
        plot_tree(clf, feature_names=X_binned.columns, filled=True)
        plt.show()
    return clf

# ---------------------------
# A7: Decision Boundary
# ---------------------------
def plot_decision_boundary_two_features(X_df, y, featA, featB):
    X = X_df[[featA, featB]].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(Xtr, ytr)
    if ENABLE_PLOTS:
        x_min, x_max = X[:,0].min(), X[:,0].max()
        y_min, y_max = X[:,1].min(), X[:,1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(Xtr[:,0], Xtr[:,1], c=ytr, label="Train", edgecolor="k")
        plt.scatter(Xte[:,0], Xte[:,1], c=yte, marker="^", s=60, label="Test", edgecolor="k")
        plt.xlabel(featA); plt.ylabel(featB); plt.legend(); plt.show()
    print("Confusion Matrix:\n", confusion_matrix(yte, clf.predict(Xte)))
    print(classification_report(yte, clf.predict(Xte), digits=3))

# ---------------------------
# Main run for one file
# ---------------------------
def main_run(psg_file, hyp_file):
    raw = load_raw(psg_file, hyp_file)
    ch = pick_best_channel(raw)
    print("Picked channel:", ch)

    # Try N2 vs REM
    df = extract_epoch_features(raw, keep_stages=(2,4))
    if df.shape[0] == 0:
        print("⚠️ No N2/REM found. Falling back to all stages.")
        df = extract_epoch_features(raw, keep_stages=None)

    print("Extracted epochs:", df.shape[0])
    print("Unique stages found:", df["stage"].unique())
    if df.shape[0] == 0:
        print("❌ Still no usable epochs. Skipping file.")
        return

    # A1
    print("\n[A1] Entropy:", entropy(df["stage"]))
    codes_w, edges_w = equal_width_binning(df["mean"].values)
    print("[A1] Equal-width bin edges:", edges_w)

    # A2
    print("\n[A2] Gini index:", gini_index(df["stage"]))

    # A3/A4
    best, ig_scores, binned = choose_root_attribute(df.drop(columns=["stage"]), df["stage"])
    print("\n[A3/A4] IG scores:", ig_scores)
    print("Best root attribute:", best)

    # A5
    tree = build_id3_tree(binned, df["stage"])
    preds = predict_id3_batch(tree, binned)
    print("\n[A5] Confusion Matrix:\n", confusion_matrix(df["stage"], preds))

    # A6
    print("\n[A6] Sklearn Decision Tree:")
    sklearn_tree_visualization(binned, df["stage"])

    # A7
    print("\n[A7] Decision Boundary (mean vs std):")
    plot_decision_boundary_two_features(df, df["stage"], "mean", "std")

# ---------------------------
# Loop through dataset
# ---------------------------
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
                print("\n=== Running on:", psg, "===")
                main_run(psg, hyp)
                count += 1
                if max_files and count >= max_files:
                    return

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    run_on_dataset(DATA_PATH, max_files=1)   # change to None for all
