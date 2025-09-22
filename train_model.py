# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings


# Load dataset
df = pd.read_csv("ckd_preprocessed.csv")

# Features and target
X = df.drop(columns=["class"])
y = df["class"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Handle imbalance only on training set ---
smote = SMOTE(sampling_strategy=0.7, random_state=42)  # partial balance
X_res, y_res = smote.fit_resample(X_train, y_train)

# --- Scaling ---
scaler = StandardScaler()
X_res_scaled = pd.DataFrame(
    scaler.fit_transform(X_res),
    columns=X_res.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# --- Models ---
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=20,
    min_child_samples=10,
    max_depth=-1,
    verbosity=-1,
    random_state=42
)

# --- Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_scores = cross_val_score(rf_model, X_res_scaled, y_res, cv=cv, scoring="accuracy")
lgb_scores = cross_val_score(lgb_model, X_res_scaled, y_res, cv=cv, scoring="accuracy")

print("=== Random Forest CV Results ===")
print("Accuracy per fold:", rf_scores)
print("Mean Accuracy:", np.mean(rf_scores))

print("\n=== LightGBM CV Results ===")
print("Accuracy per fold:", lgb_scores)
print("Mean Accuracy:", np.mean(lgb_scores))

# --- Final training & Test Evaluation ---
rf_model.fit(X_res_scaled, y_res)
lgb_model.fit(X_res_scaled, y_res)

rf_test_acc = rf_model.score(X_test_scaled, y_test)
lgb_test_acc = lgb_model.score(X_test_scaled, y_test)

print("\n=== Final Test Results ===")
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")
print(f"LightGBM Test Accuracy: {lgb_test_acc:.4f}")

# --- Save models and scaler ---
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(lgb_model, "models/lightgbm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Models and scaler saved successfully!")
