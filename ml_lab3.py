import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mne
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import minkowski

# Step 1: Set file paths
psg_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf'
hyp_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf'

# Step 2: Load EDF files
raw = mne.io.read_raw_edf(psg_file, preload=True)
raw.pick_channels(['EEG Fpz-Cz'])

# Step 3: Load and attach annotations
annotations = mne.read_annotations(hyp_file)
raw.set_annotations(annotations)

# Step 4: Map annotations to events
event_dict = {
    'Sleep stage R': 5,
    'Sleep stage 2': 2,
}
events, _ = mne.events_from_annotations(raw, event_id=event_dict)

# Step 5: Create 30s epochs using these events only
tmin = 0
tmax = 30 - 1.0 / raw.info['sfreq']
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True)

# Step 6: Get data and labels
X = epochs.get_data()
X = X.reshape(X.shape[0], -1)  # flatten

y = epochs.events[:, -1]  # labels: 2 or 5
le = LabelEncoder()
y = le.fit_transform(y)  # Convert to 0 and 1 (0=N2, 1=REM)

# A1: Centroid distance and spread
class_0 = X[y == 0]
class_1 = X[y == 1]
centroid_0 = np.mean(class_0, axis=0)
centroid_1 = np.mean(class_1, axis=0)
spread_0 = np.std(class_0, axis=0)
spread_1 = np.std(class_1, axis=0)
distance = np.linalg.norm(centroid_0 - centroid_1)
print("Centroid distance between N2 and REM:", distance)

# A2: Histogram of one feature
feature = X[:, 100]  # arbitrary index
plt.hist(feature, bins=30, color='skyblue')
plt.title("Histogram of Feature 100")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
print("Mean:", np.mean(feature), "Variance:", np.var(feature))

# A3: Minkowski distance plot
vec1 = X[0]
vec2 = X[1]
distances = [minkowski(vec1, vec2, p=r) for r in range(1, 11)]
plt.plot(range(1, 11), distances, marker='o')
plt.title("Minkowski Distance vs r")
plt.xlabel("r (Order)")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# A4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# A5: Train kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# A6: Accuracy
accuracy = knn.score(X_test, y_test)
print("kNN Accuracy (k=3):", accuracy)

# A7: Predictions
preds = knn.predict(X_test)
print("Sample Predictions:", preds[:10])

# A8: Accuracy vs. different k values
k_vals = range(1, 12)
acc_vals = []
for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc_vals.append(model.score(X_test, y_test))
plt.plot(k_vals, acc_vals, marker='s')
plt.title("kNN Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# A9: Confusion matrix and evaluation metrics
cm = confusion_matrix(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("Confusion Matrix:\n", cm)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
