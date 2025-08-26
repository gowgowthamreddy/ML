import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)


# =====================================================
# A1: Sleep Stage Classification (N2 vs REM) with kNN
# =====================================================

def load_sleep_data(psg_file, hyp_file, target_channels=['EEG Fpz-Cz']):
    raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None)
    annotations = mne.read_annotations(hyp_file)
    raw.set_annotations(annotations)
    raw.pick_channels(target_channels)
    return raw


def extract_features(raw):
    mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4
    }

    sfreq = int(raw.info['sfreq'])
    signal = raw.get_data()[0]
    epochs, labels = [], []

    for ann in raw.annotations:
        onset = int(ann['onset'] * sfreq)
        label = mapping.get(ann['description'], -1)
        if label in [2, 4]:  # N2 and REM
            segment = signal[onset:onset + 30 * sfreq]
            if len(segment) == 30 * sfreq:
                features = [
                    np.mean(segment), np.std(segment),
                    np.min(segment), np.max(segment),
                    np.percentile(segment, 25), np.percentile(segment, 75)
                ]
                epochs.append(features)
                labels.append(0 if label == 2 else 1)
    return np.array(epochs), np.array(labels)


def train_knn_classifier(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_classifier(y_train, y_test, y_train_pred, y_test_pred):
    def report(name, y_true, y_pred):
        print(f"{name} Performance:")
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print(f"Precision: {precision_score(y_true, y_pred):.3f}")
        print(f"Recall: {recall_score(y_true, y_pred):.3f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.3f}\n")

    report("Training Set", y_train, y_train_pred)
    report("Testing Set", y_test, y_test_pred)


# =====================================================
# A2: Purchase Data Regression with kNN
# =====================================================

def read_purchase_data(path):
    return pd.read_excel(path, sheet_name='Purchase_data')


def split_and_train_regressor(df, k=3):
    features = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    labels = df['Payment (Rs)']

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_regressor(y_test, y_test_pred):
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_test_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")


# =====================================================
# A3: Generate Synthetic Data
# =====================================================

def generate_synthetic_data(num_points=20, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(1, 10, num_points)
    Y = np.random.uniform(1, 10, num_points)
    classes = np.where(X + Y > 11, 'class1 - Red', 'class0 - Blue')
    data = pd.DataFrame({'X': X, 'Y': Y, 'Class': classes})
    return data


def plot_data(data):
    plt.figure(figsize=(8, 6))
    colors = {'class0 - Blue': 'blue', 'class1 - Red': 'red'}
    plt.scatter(data['X'], data['Y'], c=data['Class'].map(colors))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Training Data by Class')
    plt.grid(True)
    plt.show()
    return data


# =====================================================
# A4–A5: kNN Decision Boundaries
# =====================================================

def plot_knn_decision_boundary(data, k=3, step=0.1):
    x_min, x_max, y_min, y_max = 0, 10, 0, 10
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )
    test_points = np.c_[xx.ravel(), yy.ravel()]
    X_train = data[['X', 'Y']]
    y_train = data['Class']

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    Z = knn.predict(test_points)
    test_data = pd.DataFrame({
        'X': test_points[:, 0],
        'Y': test_points[:, 1],
        'Predicted_Class': Z
    })

    plt.figure(figsize=(10, 8))
    colors = {'class0 - Blue': 'blue', 'class1 - Red': 'red'}
    plt.scatter(test_data['X'], test_data['Y'],
                c=test_data['Predicted_Class'].map(colors), s=5, alpha=0.5)
    plt.scatter(data['X'], data['Y'], c=data['Class'].map(colors),
                edgecolors='k', s=50, label='Training Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'kNN (k={k}) Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# A6: Analyze kNN Performance for Different k
# =====================================================

def analyze_knn_performance(X_train, y_train, X_test, y_test, k_values):
    train_f1, test_f1 = [], []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        f1_train = f1_score(y_train, knn.predict(X_train))
        f1_test = f1_score(y_test, knn.predict(X_test))

        train_f1.append(f1_train)
        test_f1.append(f1_test)

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, train_f1, marker='o', label='Train F1 Score')
    plt.plot(k_values, test_f1, marker='s', label='Test F1 Score')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1 Score")
    plt.title("kNN Performance vs. Value of k (N2 vs REM)")
    plt.legend()
    plt.grid(True)
    plt.show()


# =====================================================
# A7: Grid Search for Best k
# =====================================================

def tune_knn_hyperparameters(X, y, max_k=20):
    param_grid = {'n_neighbors': np.arange(1, max_k + 1)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"Best k value: {grid_search.best_params_['n_neighbors']}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Best kNN model: {grid_search.best_estimator_}")
    return grid_search.best_estimator_


# =====================================================
# === DRIVER CODE (Run everything as before) ===
# =====================================================

if _name_ == "_main_":
    # --- A1 ---
    psg_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf'
    hyp_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf'
    raw = load_sleep_data(psg_file, hyp_file)
    X, y = extract_features(raw)
    model, X_train, X_test, y_train, y_test = train_knn_classifier(X, y)
    evaluate_classifier(y_train, y_test,
                        model.predict(X_train), model.predict(X_test))

    # --- A2 ---
    path1 = "/content/LabSessionData.xlsx"
    df1 = read_purchase_data(path1)
    reg_model, X_train1, X_test1, y_train1, y_test1 = split_and_train_regressor(df1)
    y_test_pred1 = reg_model.predict(X_test1)
    evaluate_regressor(y_test1, y_test_pred1)

    # --- A3 ---
    data = generate_synthetic_data()
    plot_data(data)

    # --- A4 & A5 ---
    plot_knn_decision_boundary(data, k=3)
    plot_knn_decision_boundary(data, k=1)  # change k to see differences

    # --- A6 ---
    analyze_knn_performance(X_train, y_train, X_test, y_test, k_values=list(range(1, 16)))

    # --- A7 ---
    tune_knn_hyperparameters(X, y)
