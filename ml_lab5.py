import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import mne
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =====================================================
# Utility Functions
# =====================================================

def find_channel_name(raw, candidates=None):
    if candidates is None:
        candidates = ['Fpz-Cz', 'EEG Fpz-Cz', 'EEG Fpz-Cz (referential)',
                      'Fpz-Cz (Cz)', 'FpzCz']
    for c in candidates:
        if c in raw.info['ch_names']:
            return c
    for ch in raw.info['ch_names']:
        if 'fpz' in ch.lower():
            return ch
    for ch in raw.info['ch_names']:
        if 'eeg' in ch.lower():
            return ch
    return None


def load_and_extract_features(psg_file, hyp_file, epoch_sec=30, verbose=True):
    if not os.path.exists(psg_file) or not os.path.exists(hyp_file):
        raise FileNotFoundError("EDF PSG or Hypnogram file not found.")

    if verbose: print("Loading PSG:", psg_file)
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose='ERROR')
    ann = mne.read_annotations(hyp_file)
    raw.set_annotations(ann)

    ch = find_channel_name(raw)
    if not ch:
        raise ValueError("No suitable EEG channel found (e.g., Fpz-Cz).")
    if verbose: print("Selected channel:", ch)

    raw_pick = raw.copy().pick_channels([ch])
    sfreq = raw_pick.info['sfreq']
    signal = raw_pick.get_data()[0]

    mapping = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
        "W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4,
        "Wake": 0, "Sleep stage ?": -1
    }

    feats, labels, descs = [], [], []
    for a in raw.annotations:
        desc = a['description']
        onset, dur = int(a['onset'] * sfreq), int(epoch_sec * sfreq)
        end = onset + dur
        if end > len(signal): continue

        label = mapping.get(desc, -1)
        if label == -1 and desc.isdigit():
            label = int(desc)

        seg = signal[onset:end]
        feats.append([np.mean(seg), np.std(seg), np.min(seg), np.max(seg),
                      np.percentile(seg, 25), np.percentile(seg, 75)])
        labels.append(label)
        descs.append(desc)

    df = pd.DataFrame(feats, columns=['feat_mean','feat_std','feat_min','feat_max','feat_p25','feat_p75'])
    df['stage'] = labels
    df['stage_str'] = descs
    if verbose: print(f"Extracted {len(df)} epochs, features = {df.shape[1]-2}")
    return df


def print_regression_metrics(y_train, y_train_pred, y_test, y_test_pred):
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (np.where(y_true==0, 1e-8, y_true)))) * 100
    print("Train -> MSE:", mean_squared_error(y_train, y_train_pred),
          "RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)),
          "MAPE:", mape(y_train, y_train_pred),
          "R2:", r2_score(y_train, y_train_pred))
    print("Test  -> MSE:", mean_squared_error(y_test, y_test_pred),
          "RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)),
          "MAPE:", mape(y_test, y_test_pred),
          "R2:", r2_score(y_test, y_test_pred))


# =====================================================
# Regression Tasks (A1, A2, A3)
# =====================================================

def single_attribute_regression(train_df, test_df):
    print("\n=== A1: Linear Regression (Single Attribute) ===")
    X_train, y_train = train_df[['f1']].values, train_df['target_mean'].values
    X_test, y_test = test_df[['f1']].values, test_df['target_mean'].values
    reg = LinearRegression().fit(X_train, y_train)
    print_regression_metrics(y_train, reg.predict(X_train), y_test, reg.predict(X_test))


def multi_attribute_regression(train_df, test_df):
    print("\n=== A3: Linear Regression (Multi-Attribute) ===")
    X_train, y_train = train_df[['f0','f1','f2','f3','f4','f5']], train_df['target_mean']
    X_test, y_test = test_df[['f0','f1','f2','f3','f4','f5']], test_df['target_mean']
    reg = LinearRegression().fit(X_train, y_train)
    print_regression_metrics(y_train, reg.predict(X_train), y_test, reg.predict(X_test))


# =====================================================
# Clustering Tasks (A4–A7)
# =====================================================

def baseline_kmeans(X, k=2):
    print(f"\n=== A4: Baseline KMeans (k={k}) ===")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    labels = kmeans.labels_
    print("Cluster centers (scaled):\n", kmeans.cluster_centers_)
    print(f"Silhouette: {silhouette_score(X, labels):.4f}, "
          f"CH: {calinski_harabasz_score(X, labels):.4f}, "
          f"DB: {davies_bouldin_score(X, labels):.4f}")
    return kmeans


def evaluate_kmeans_range(X, ks=range(2,13)):
    print("\n=== A6: KMeans Evaluation (k=2..12) ===")
    sil, ch, db, inertias = [], [], [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        labs = km.labels_
        sil.append(silhouette_score(X, labs))
        ch.append(calinski_harabasz_score(X, labs))
        db.append(davies_bouldin_score(X, labs))
        inertias.append(km.inertia_)

    # Plots
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.plot(ks, sil, marker='o'); plt.title("Silhouette vs k")
    plt.subplot(1,3,2); plt.plot(ks, ch, marker='o'); plt.title("CH vs k")
    plt.subplot(1,3,3); plt.plot(ks, db, marker='o'); plt.title("DB vs k")
    plt.tight_layout(); plt.show()

    plt.figure(); plt.plot(ks, inertias, marker='o'); plt.title("Elbow Plot")
    plt.xlabel("k"); plt.ylabel("Inertia"); plt.grid(True); plt.show()

    print("Best k (Silhouette):", ks[np.argmax(sil)])
    print("Best k (CH):", ks[np.argmax(ch)])
    print("Best k (DB):", ks[np.argmin(db)])
    return inertias


# =====================================================
# === DRIVER CODE ===
# =====================================================

if _name_ == "_main_":
    psg_file = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf"
    hyp_file = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf"

    df = load_and_extract_features(psg_file, hyp_file, epoch_sec=30, verbose=True)
    df = df[df['stage'] != -1].reset_index(drop=True)

    # Standardize
    X = StandardScaler().fit_transform(df[['feat_mean','feat_std','feat_min','feat_max','feat_p25','feat_p75']])
    df_feats = pd.DataFrame(X, columns=['f0','f1','f2','f3','f4','f5'])
    df_feats['target_mean'] = df['feat_mean'].values

    n_train = int(0.7*len(df_feats))
    train_df, test_df = df_feats.iloc[:n_train], df_feats.iloc[n_train:]

    # Regression tasks
    single_attribute_regression(train_df, test_df)
    multi_attribute_regression(train_df, test_df)

    # Clustering tasks
    baseline_kmeans(X, k=2)
    evaluate_kmeans_range(X)
