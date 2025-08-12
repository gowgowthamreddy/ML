
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import mne
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


psg_file = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf"
hyp_file = r"C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf"
# -----------------------------------

def find_channel_name(raw, candidates=None):
    # try common Fpz-Cz variants
    if candidates is None:
        candidates = ['Fpz-Cz', 'EEG Fpz-Cz', 'EEG Fpz-Cz (referential)', 'Fpz-Cz (Cz)', 'FpzCz']
    chs = raw.info['ch_names']
    for c in candidates:
        if c in chs:
            return c
    # fallback: try partial match containing 'Fpz' or 'Fpz'+'Cz'
    for ch in chs:
        if 'Fpz' in ch or 'FPZ' in ch or 'fpz' in ch:
            return ch
    # if still nothing, return first EEG channel if available
    for ch in chs:
        if ch.lower().startswith('eeg') or 'eeg' in ch.lower():
            return ch
    # else None
    return None

def load_and_extract_features(psg_file, hyp_file, epoch_sec=30, selected_stage_map=None, verbose=True):
    """
    Loads 1 EDF PSG and its hypnogram, extracts simple time-domain features per 30s epoch.
    Returns pandas.DataFrame with columns: feat0..feat5, stage (int), stage_str
    """
    if not os.path.exists(psg_file):
        raise FileNotFoundError(f"PSG file not found: {psg_file}")
    if not os.path.exists(hyp_file):
        raise FileNotFoundError(f"Hypnogram file not found: {hyp_file}")

    if verbose:
        print("Loading PSG EDF:", psg_file)
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None, verbose='ERROR')

    # Read annotations (hypnogram)
    if verbose:
        print("Loading hypnogram:", hyp_file)
    ann = mne.read_annotations(hyp_file)
    raw.set_annotations(ann)

    # pick channel
    ch_name = find_channel_name(raw)
    if ch_name is None:
        raise ValueError("Couldn't find a suitable EEG channel (e.g., Fpz-Cz) in EDF.")
    if verbose:
        print("Selected channel:", ch_name)

    # pick and get data
    raw_pick = raw.copy().pick_channels([ch_name])
    sfreq = raw_pick.info['sfreq']
    signal = raw_pick.get_data()[0]  # 1D array

    # mapping strings in annotation descriptions to numeric labels (common Sleep-EDF labels)
    # many hypnograms use these exact strings; adjust if yours differs
    mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,  # collapse stage 3 & 4 if present
        "Sleep stage R": 4,
        "W": 0, "1": 1, "2": 2, "3": 3, "4":3, "R": 4,
        "Wake": 0, "Sleep stage ?": -1
    }

    # Convert annotations -> epochs of epoch_sec seconds
    epochs_features = []
    epochs_labels = []
    epochs_stage_str = []

    for a in raw.annotations:
        desc = a['description']
        onset_s = a['onset']          # seconds
        duration_s = a['duration']    # seconds (should be 30)
        label = mapping.get(desc, None)
        # some hypnograms use numeric textual '2' etc. Try parse
        if label is None:
            try:
                if isinstance(desc, str) and desc.strip().isdigit():
                    label = int(desc.strip())
                else:
                    # not recognized
                    label = -1
            except:
                label = -1
        # only use fixed-length epochs of epoch_sec
        start_idx = int(onset_s * sfreq)
        end_idx = start_idx + int(epoch_sec * sfreq)
        if end_idx > len(signal):
            # skip epochs that don't fully fit
            continue
        seg = signal[start_idx:end_idx]
        # compute 6 time-domain features (same as earlier lab)
        feats = [
            np.mean(seg),
            np.std(seg),
            np.min(seg),
            np.max(seg),
            np.percentile(seg, 25),
            np.percentile(seg, 75)
        ]
        epochs_features.append(feats)
        epochs_labels.append(label)
        epochs_stage_str.append(desc)

    df = pd.DataFrame(epochs_features, columns=['feat_mean','feat_std','feat_min','feat_max','feat_p25','feat_p75'])
    df['stage'] = epochs_labels
    df['stage_str'] = epochs_stage_str

    if verbose:
        print(f"Extracted {len(df)} epochs, feature vector length = {df.shape[1]-2}")
    return df

# ---------- Run feature extraction ----------
try:
    df = load_and_extract_features(psg_file, hyp_file, epoch_sec=30, verbose=True)
except Exception as e:
    raise RuntimeError(f"Something went wrong loading files: {e}")

# Prepare dataset: drop invalid stage (-1) if any
df = df[df['stage'] != -1].reset_index(drop=True)
# For many lab tasks use all epochs (regression/clustering uses feature columns)
X_all = df[['feat_mean','feat_std','feat_min','feat_max','feat_p25','feat_p75']].values
y_stage = df['stage'].values

# Standardize features for clustering/regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Train/test split for regression tasks
X_train, X_test = X_scaled[:int(0.7*len(X_scaled))], X_scaled[int(0.7*len(X_scaled)):]
# For "targets" we will define synthetic numeric targets based on features (instructions allow)
# A1: Single attribute regression -> predict feat_mean (as target) using feat_std (single feature)
# A3: Multi-attribute regression -> predict feat_mean using other features

# build DataFrames for clarity
df_feats = pd.DataFrame(X_scaled, columns=['f0','f1','f2','f3','f4','f5'])
df_feats['target_mean'] = df['feat_mean'].values  # original unscaled mean used as target
# split indices
n_train = int(0.7*len(df_feats))
train_df = df_feats.iloc[:n_train].reset_index(drop=True)
test_df = df_feats.iloc[n_train:].reset_index(drop=True)

# ---------------- A1: Single-attribute Linear Regression ----------------
print("\n=== A1: Linear Regression (single attribute) ===")
# X: single feature (use f1 scaled), y: target_mean (original-scale target)
X1_train = train_df[['f1']].values
y1_train = train_df['target_mean'].values
X1_test = test_df[['f1']].values
y1_test = test_df['target_mean'].values

reg1 = LinearRegression()
reg1.fit(X1_train, y1_train)
y1_train_pred = reg1.predict(X1_train)
y1_test_pred = reg1.predict(X1_test)

def print_regression_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (np.where(y_true==0, 1e-8, y_true)))) * 100
    print("Train MSE, RMSE, MAPE, R2:",
          mean_squared_error(y_true_train, y_pred_train),
          np.sqrt(mean_squared_error(y_true_train, y_pred_train)),
          mape(y_true_train, y_pred_train),
          r2_score(y_true_train, y_pred_train))
    print("Test  MSE, RMSE, MAPE, R2:",
          mean_squared_error(y_true_test, y_pred_test),
          np.sqrt(mean_squared_error(y_true_test, y_pred_test)),
          mape(y_true_test, y_pred_test),
          r2_score(y_true_test, y_pred_test))

print_regression_metrics(y1_train, y1_train_pred, y1_test, y1_test_pred)

# ---------------- A2: Metrics already printed above (Train/Test MSE RMSE MAPE R2) ----------------

# ---------------- A3: Multi-attribute Linear Regression ----------------
print("\n=== A3: Linear Regression (multi-attribute) ===")
# predict target_mean using all other standardized features (f0..f5)
X3_train = train_df[['f0','f1','f2','f3','f4','f5']].values
y3_train = train_df['target_mean'].values
X3_test = test_df[['f0','f1','f2','f3','f4','f5']].values
y3_test = test_df['target_mean'].values

reg3 = LinearRegression()
reg3.fit(X3_train, y3_train)
y3_train_pred = reg3.predict(X3_train)
y3_test_pred = reg3.predict(X3_test)

print_regression_metrics(y3_train, y3_train_pred, y3_test, y3_test_pred)

# ---------------- A4: k-means clustering (k=2 baseline) ----------------
print("\n=== A4: KMeans clustering (k=2 baseline) ===")
X_cluster = X_scaled.copy()  # use standardized features
k_default = 2
kmeans_default = KMeans(n_clusters=k_default, random_state=42, n_init="auto").fit(X_cluster)
labels_default = kmeans_default.labels_
centers_default = kmeans_default.cluster_centers_
print("Cluster centers (scaled):\n", centers_default)

sil = silhouette_score(X_cluster, labels_default)
ch = calinski_harabasz_score(X_cluster, labels_default)
db = davies_bouldin_score(X_cluster, labels_default)
print(f"Silhouette (k=2): {sil:.4f}, Calinski-Harabasz: {ch:.4f}, Davies-Bouldin: {db:.4f}")

# ---------------- A5: compute the three metrics for k=2 (done above) ----------------

# ---------------- A6: Vary k and plot Silhouette, CH, DB ----------------
print("\n=== A6: Evaluate KMeans for k=2..12 and plot metrics ===")
ks = list(range(2, 13))
sil_scores = []
ch_scores = []
db_scores = []
inertias = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_cluster)
    labs = kmeans.labels_
    sil_scores.append(silhouette_score(X_cluster, labs))
    ch_scores.append(calinski_harabasz_score(X_cluster, labs))
    db_scores.append(davies_bouldin_score(X_cluster, labs))
    inertias.append(kmeans.inertia_)

# Plotting
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.plot(ks, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k"); plt.ylabel("Silhouette")

plt.subplot(1,3,2)
plt.plot(ks, ch_scores, marker='o')
plt.title("Calinski-Harabasz vs k")
plt.xlabel("k"); plt.ylabel("CH Score")

plt.subplot(1,3,3)
plt.plot(ks, db_scores, marker='o')
plt.title("Davies-Bouldin vs k")
plt.xlabel("k"); plt.ylabel("DB Index")
plt.tight_layout()
plt.show()

# ---------------- A7: Elbow plot for inertia vs k ----------------
plt.figure(figsize=(6,4))
plt.plot(ks, inertias, marker='o')
plt.title("Elbow plot (Inertia) vs k")
plt.xlabel("k"); plt.ylabel("Inertia (distortion)")
plt.grid(True)
plt.show()

# Suggest best k by each metric
best_k_sil = ks[int(np.argmax(sil_scores))]
best_k_ch = ks[int(np.argmax(ch_scores))]
best_k_db = ks[int(np.argmin(db_scores))]
best_k_elbow = ks[int(np.argmin(np.gradient(inertias)))]  # heuristic: where gradient smallest

print("Suggested best k by metrics:")
print(" Best Silhouette k:", best_k_sil)
print(" Best Calinski-Harabasz k:", best_k_ch)
print(" Best Davies-Bouldin k:", best_k_db)


print("\nInertias (k=2..12):")
for k, val in zip(ks, inertias):
    print(f" k={k} -> inertia={val:.4e}")

