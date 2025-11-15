import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ==========================================================
#                 AUTO-DETECT PROJECT PATH
# ==========================================================
# Path to /notebooks folder → go one level up
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "final_model"
MODEL_DIR.mkdir(exist_ok=True)

print("Base Directory:", BASE_DIR)
print("Data Directory:", DATA_DIR)

# ==========================================================
#                  LOAD DATASETS SAFELY
# ==========================================================
train_path = DATA_DIR / "cleaned_train.csv"
test_path = DATA_DIR / "cleaned_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# ==========================================================
#                     SPLIT FEATURES
# ==========================================================
X_train = train_df.drop(['subject', 'Activity'], axis=1)
y_train = train_df['Activity']

X_test = test_df.drop(['subject', 'Activity'], axis=1)
y_test = test_df['Activity']

# ==========================================================
#                     SCALING
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
#                     PCA (99% variance → 179 components)
# ==========================================================
pca = PCA(n_components=179)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA Components: {pca.n_components_}")
print(f"Variance Explained: {pca.explained_variance_ratio_.sum():.4f}")

# ==========================================================
#                  GRID SEARCH SVM (optimized)
# ==========================================================
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],    # Best performing & fastest
    'gamma': ['scale']
}

svm = SVC(random_state=42, probability=True)

grid_search_pca = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=3
)

print("\n===== Running Grid Search (PCA + SVM) =====")
grid_search_pca.fit(X_train_pca, y_train)

print("\n===== PCA + SVM Results =====")
print("Best Parameters :", grid_search_pca.best_params_)
print("Best CV Accuracy:", grid_search_pca.best_score_)

# ==========================================================
#              TRAIN BEST MODEL ON FULL DATA
# ==========================================================
best_svm_pca = grid_search_pca.best_estimator_
best_svm_pca.fit(X_train_pca, y_train)

# ==========================================================
#              EVALUATION ON TEST SET
# ==========================================================
y_pred = best_svm_pca.predict(X_test_pca)

print("\n===== Evaluation (Test Set) =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

# ==========================================================
#              SAVE MODELS FOR STREAMLIT
# ==========================================================
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(pca, MODEL_DIR / "pca.pkl")
joblib.dump(best_svm_pca, MODEL_DIR / "svm_pca.pkl")

print("\n===== Saved Models =====")
print(MODEL_DIR / "scaler.pkl")
print(MODEL_DIR / "pca.pkl")
print(MODEL_DIR / "svm_pca.pkl")
print("🚀 Training Complete!")
