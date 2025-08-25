import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mne
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import minkowski


# ----------------------------
# Data Loading and Preprocessing
# ----------------------------
def load_data(psg_file, hyp_file, channel='EEG Fpz-Cz'):
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    raw.pick_channels([channel])
    annotations = mne.read_annotations(hyp_file)
    raw.set_annotations(annotations)
    return raw


def create_epochs(raw, event_dict, epoch_length=30):
    events, _ = mne.events_from_annotations(raw, event_id=event_dict)
    tmin = 0
    tmax = epoch_length - 1.0 / raw.info['sfreq']
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    return epochs


def extract_data_labels(epochs):
    X = epochs.get_data().reshape(epochs.get_data().shape[0], -1)
    y = epochs.events[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


# ----------------------------
# Feature Analysis
# ----------------------------
def analyze_centroids(X, y):
    class_0, class_1 = X[y == 0], X[y == 1]
    centroid_0, centroid_1 = np.mean(class_0, axis=0), np.mean(class_1, axis=0)
    distance = np.linalg.norm(centroid_0 - centroid_1)
    print("Centroid distance between N2 and REM:", distance)


def plot_histogram(X, feature_idx=100):
    feature = X[:, feature_idx]
    plt.hist(feature, bins=30, color='skyblue')
    plt.title(f"Histogram of Feature {feature_idx}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    print("Mean:", np.mean(feature), "Variance:", np.var(feature))


def plot_minkowski(X, idx1=0, idx2=1):
    vec1, vec2 = X[idx1], X[idx2]
    distances = [minkowski(vec1, vec2, p=r) for r in range(1, 11)]
    plt.plot(range(1, 11), distances, marker='o')
    plt.title("Minkowski Distance vs r")
    plt.xlabel("r (Order)")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()


# ----------------------------
# Model Training & Evaluation
# ----------------------------
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("kNN Accuracy (k=3):", accuracy)
    print("Sample Predictions:", preds[:10])
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    return accuracy


def plot_accuracy_vs_k(X_train, y_train, X_test, y_test, k_range=range(1, 12)):
    acc_vals = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc_vals.append(model.score(X_test, y_test))

    plt.plot(k_range, acc_vals, marker='s')
    plt.title("kNN Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


# ----------------------------
# Main Execution
# ----------------------------
def main():
    # File paths
    psg_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf'
    hyp_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf'

    # Events mapping
    event_dict = {
        'Sleep stage R': 5,
        'Sleep stage 2': 2,
    }

    # Pipeline
    raw = load_data(psg_file, hyp_file)
    epochs = create_epochs(raw, event_dict)
    X, y = extract_data_labels(epochs)

    # Feature Analysis
    analyze_centroids(X, y)
    plot_histogram(X, feature_idx=100)
    plot_minkowski(X)

    # Train-test split (with fixed seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate model
    knn = train_knn(X_train, y_train, k=3)
    evaluate_model(knn, X_test, y_test)

    # Accuracy vs k
    plot_accuracy_vs_k(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()

