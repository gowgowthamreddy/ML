# A1

import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load PSG and hypnogram files
psg_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf'
hyp_file = r'C:\Users\asus\Downloads\sleep-edf-database-expanded-1.0.0\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf'

# Load and prepare data
raw = mne.io.read_raw_edf(psg_file, preload=True, stim_channel=None)
annotations = mne.read_annotations(hyp_file)
raw.set_annotations(annotations)


raw.pick_channels(['EEG Fpz-Cz'])

# Map sleep stage labels
mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

# Segment EEG into 30s epochs and extract N2 & REM
sfreq = int(raw.info['sfreq'])
signal = raw.get_data()[0]
epochs, labels = [], []

for ann in raw.annotations:
    onset = int(ann['onset'] * sfreq)
    label = mapping.get(ann['description'], -1)
    if label in [2, 4]:  # N2 and REM only
        segment = signal[onset:onset + 30 * sfreq]
        if len(segment) == 30 * sfreq:
            features = [
                np.mean(segment), np.std(segment),
                np.min(segment), np.max(segment),
                np.percentile(segment, 25), np.percentile(segment, 75)
            ]
            epochs.append(features)
            labels.append(0 if label == 2 else 1)  # N2=0, REM=1

X = np.array(epochs)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train classifier
n = KNeighborsClassifier(n_neighbors=3)
n.fit(X_train, y_train)

# Predictions
y_train_pred = n.predict(X_train)
y_test_pred = n.predict(X_test)

# Classification Metrics
def performance_metrics(y_train, y_test, y_train_pred, y_test_pred):
    print("Training Set Performance:")
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    print("\nTesting Set Performance:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

# Show results
performance_metrics(y_train, y_test, y_train_pred, y_test_pred)

# A2
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor # Use KNeighborsRegressor for regression

def Read_data(path):
  #read data from excel
  df1 = pd.read_excel(path, sheet_name='Purchase_data')
  return df1

def test_train_data(df1):
  #split train and test data for price pridiction
  features = df1[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
  labels = df1['Payment (Rs)'] # Target variable is 'Payment (Rs)'

  X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

  # Define and train a new KNeighborsRegressor for this dataset
  n1 = KNeighborsRegressor(n_neighbors=3) # Use KNeighborsRegressor for regression
  n1.fit(X_train, y_train)

  return (X_train, X_test, y_train, y_test, n1)

def do_prediction(n_model, X_train, X_test):
  y_train_pred = n_model.predict(X_train)
  y_test_pred = n_model.predict(X_test)
  return (y_train_pred, y_test_pred)

def operations(y_train, y_test, y_train_pred, y_test_pred):
  # Calculate MSE
  mse = mean_squared_error(y_test, y_test_pred)
  print(f"Mean Squared Error (MSE): {mse:.4f}")

  # Calculate RMSE
  rmse = np.sqrt(mse)
  print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

  # Calculate MAPE
  # Handle potential division by zero if y_test contains 0
  # Adding a small epsilon to avoid division by zero in MAPE
  mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-8))) * 100
  print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")


  # Calculate R2 score
  r2 = r2_score(y_test, y_test_pred)
  print(f"R-squared (R2) Score: {r2:.4f}")


path1 = "/content/LabSessionData.xlsx"
df1 = Read_data(path1)
X_train1, X_test1, y_train1, y_test1, n1 = test_train_data(df1)
y_train_pred1, y_test_pred1 = do_prediction(n1, X_train1, X_test1)
operations(y_train1, y_test1, y_train_pred1, y_test_pred1)

# A3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate Data
np.random.seed(42) # for reproducibility
num_points = 20
X = np.random.uniform(1, 10, num_points)
Y = np.random.uniform(1, 10, num_points)

# 2. Assign Classes (example condition: if X + Y > 11, assign to class1, otherwise class0)
classes = np.where(X + Y > 11, 'class1 - Red', 'class0 - Blue')

# 3. Create DataFrame
data = pd.DataFrame({'X': X, 'Y': Y, 'Class': classes})

# 4. Visualize Data
plt.figure(figsize=(8, 6))
colors = {'class0 - Blue': 'blue', 'class1 - Red': 'red'}
plt.scatter(data['X'], data['Y'], c=data['Class'].map(colors))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data by Class')
plt.grid(True)

# 5. Observe Plot
plt.show()

# 6. Finish task - Describe the generated data and the plot
print("Generated Data:")
display(data)

print("\nObservation of the plot:")
print("The scatter plot shows 20 data points with two features, X and Y. The points are colored blue for 'class0' and red for 'class1'. The classes are assigned based on the condition X + Y > 11. Points where the sum of X and Y is greater than 11 are marked in red, and those where the sum is less than or equal to 11 are marked in blue. This creates a visual separation of the two classes based on a linear boundary.")

# A4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Assume 'data' DataFrame from the previous cell contains the training data

# 1. Generate Test Data
x_min, x_max = 0, 10
y_min, y_max = 0, 10
h = 0.1 # step size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Flatten the grid points for prediction
test_points = np.c_[xx.ravel(), yy.ravel()]

# Convert test_points to a DataFrame with the same column names as training data
test_data_df = pd.DataFrame(test_points, columns=['X', 'Y'])

# 2. Prepare Training Data (from the previously generated 'data' DataFrame)
X_train = data[['X', 'Y']]
y_train = data['Class']

# 3. Train kNN Classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Predict Test Data Classes
Z = knn.predict(test_data_df)

# 5. Create Test DataFrame (optional, but good for organization)
test_data = pd.DataFrame({'X': test_points[:, 0], 'Y': test_points[:, 1], 'Predicted_Class': Z})

# 6. Visualize Test Data
plt.figure(figsize=(10, 8))
colors = {'class0 - Blue': 'blue', 'class1 - Red': 'red'}
plt.scatter(test_data['X'], test_data['Y'], c=test_data['Predicted_Class'].map(colors), s=5, alpha=0.5)

# Also plot the training points to see their relation to the decision boundary
plt.scatter(data['X'], data['Y'], c=data['Class'].map(colors), edgecolors='k', s=50, label='Training Points')


plt.xlabel('X')
plt.ylabel('Y')
plt.title('kNN (k=3) Classification of Test Data with Training Points')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.grid(True)

# 7. Observe Plot
plt.show()

# 8. Finish task - Observe the color spread and class boundary lines
print("Observation of the plot:")
print("The scatter plot shows the test data points colored according to the kNN classifier's predictions. The original training points are also shown as larger points with black edges. The color spread indicates the regions in the feature space that the kNN model classifies as 'class0 - Blue' or 'class1 - Red'. The boundary between the blue and red regions represents the decision boundary learned by the kNN classifier. With k=3, the boundary is likely to be non-linear and influenced by the nearest three training points.")

# A5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Assume 'data' DataFrame from the previous cell contains the training data

# 1. Generate Test Data (same as before)
x_min, x_max = 0, 10
y_min, y_max = 0, 10
h = 0.1 # step size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Flatten the grid points for prediction
test_points = np.c_[xx.ravel(), yy.ravel()]

# Convert test_points to a DataFrame with the same column names as training data
test_data_df = pd.DataFrame(test_points, columns=['X', 'Y'])

# 2. Prepare Training Data (from the previously generated 'data' DataFrame)
X_train = data[['X', 'Y']]
y_train = data['Class']

# --- Modify the value of k here to see how the boundary changes ---
k_value = 1 # Start with k=1, you can change this value

# 3. Train kNN Classifier with the specified k
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

# 4. Predict Test Data Classes
Z = knn.predict(test_data_df)

# 5. Create Test DataFrame
test_data = pd.DataFrame({'X': test_points[:, 0], 'Y': test_points[:, 1], 'Predicted_Class': Z})

# 6. Visualize Test Data
plt.figure(figsize=(10, 8))
colors = {'class0 - Blue': 'blue', 'class1 - Red': 'red'}
plt.scatter(test_data['X'], test_data['Y'], c=test_data['Predicted_Class'].map(colors), s=5, alpha=0.5)

# Also plot the training points to see their relation to the decision boundary
plt.scatter(data['X'], data['Y'], c=data['Class'].map(colors), edgecolors='k', s=50, label='Training Points')


plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'kNN (k={k_value}) Classification of Test Data with Training Points')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.grid(True)

# 7. Observe Plot
plt.show()

# 8. Finish task - Observe the color spread and class boundary lines
print(f"Observation of the plot for k={k_value}:")
print("Observe how the decision boundary changes with the value of k. A smaller k (like 1) often results in a more complex and potentially noisy boundary, closely following individual training points. A larger k creates a smoother boundary, as it considers more neighbors, making it less susceptible to outliers but potentially blurring distinctions between classes in some areas.")

# === A6: Analyzing kNN Behavior for Different k Values ===

import matplotlib.pyplot as plt

train_f1 = []
test_f1 = []
k_values = list(range(1, 16))

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)

    train_f1.append(f1_train)
    test_f1.append(f1_test)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(k_values, train_f1, marker='o', label='Train F1 Score')
plt.plot(k_values, test_f1, marker='s', label='Test F1 Score')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("F1 Score")
plt.title("kNN Performance vs. Value of k (N2 vs REM)")
plt.legend()
plt.grid(True)
plt.show()

# A7
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Define the parameter grid for k
param_grid = {'n_neighbors': np.arange(1, 21)} # Search for k from 1 to 20

# Create a kNN classifier instance
knn = KNeighborsClassifier()

# Create GridSearchCV object
# cv=5 means 5-fold cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Print the best parameter (best k) and the corresponding best score
print(f"Best k value: {grid_search.best_params_['n_neighbors']}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# You can also access the best estimator and its parameters
best_knn_model = grid_search.best_estimator_
print(f"Best kNN model: {best_knn_model}")