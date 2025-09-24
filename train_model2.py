# train_model2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')


def preprocess_data(df):
    """
    Cleans and preprocesses the CKD dataset.
    """
    # 1. Rename columns
    column_mapping = {
        'bp': 'blood_pressure', 'sg': 'specific_gravity', 'al': 'albumin',
        'su': 'sugar', 'rbc': 'red_blood_cells', 'pc': 'pus_cell',
        'pcc': 'pus_cell_clumps', 'ba': 'bacteria', 'bgr': 'blood_glucose_random',
        'bu': 'blood_urea', 'sc': 'serum_creatinine', 'sod': 'sodium',
        'pot': 'potassium', 'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
        'wbcc': 'white_blood_cell_count', 'rbcc': 'red_blood_cell_count',
        'htn': 'hypertension', 'dm': 'diabetes_mellitus', 'cad': 'coronary_artery_disease',
        'appet': 'appetite', 'pe': 'pedal_edema', 'ane': 'anemia',
        'class': 'classification'
    }
    df.rename(columns=column_mapping, inplace=True)

    # 2. Correct data types
    numerical_cols = [
        'age', 'blood_pressure', 'blood_glucose_random', 'blood_urea',
        'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',
        'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count'
    ]
    categorical_cols = [col for col in df.columns if col not in numerical_cols]

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # strip spaces safely only for string values
    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 3. Fix inconsistent values
    df['diabetes_mellitus'].replace({'\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'].replace({'\tno': 'no', ' no': 'no'}, inplace=True)
    df['classification'].replace({'ckd\t': 'ckd'}, inplace=True)

    # 4. Fill missing values
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 5. Encode categorical features
    binary_map = {
        'normal': 1, 'abnormal': 0, 'present': 1, 'notpresent': 0,
        'yes': 1, 'no': 0, 'good': 1, 'poor': 0, 'ckd': 1, 'notckd': 0
    }

    for col in [
        'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
        'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
        'appetite', 'pedal_edema', 'anemia', 'classification'
    ]:
        df[col] = df[col].replace(binary_map)

    for col in ['specific_gravity', 'albumin', 'sugar']:
        df[col] = df[col].astype('category')

    df = pd.get_dummies(df, columns=['specific_gravity', 'albumin', 'sugar'], drop_first=True)

    # Save cleaned dataset
    df.to_csv("predataset.csv", index=False)
    print("--- Preprocessed data saved to predataset.csv ---")
    return df


def plot_confusion_and_roc(y_true, y_pred, y_proba, model_name):
    """Helper to plot confusion matrix and ROC curve."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not CKD", "CKD"],
                yticklabels=["Not CKD", "CKD"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Load dataset
    try:
        df = pd.read_csv("ckd_dataset1.csv")
    except FileNotFoundError:
        print("Error: ckd_dataset1.csv not found.")
        return

    df_clean = preprocess_data(df)

    # Features and target
    X = df_clean.drop("classification", axis=1)
    y = df_clean["classification"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE
    smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"\n--- Data ready. Resampled training shape: {X_train_resampled.shape} ---")

    # ---------------- Random Forest ----------------
    print("\n--- Training a Simplified Random Forest ---")
    best_rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        min_samples_leaf=4,
        random_state=42
    )

    best_rf.fit(X_train_resampled, y_train_resampled)

    y_pred_rf = best_rf.predict(X_test_scaled)
    y_proba_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=["Not CKD", "CKD"]))

    plot_confusion_and_roc(y_test, y_pred_rf, y_proba_rf, "Random Forest")

    # ---------------- LightGBM ----------------
    print("\n--- Training LightGBM ---")
    param_grid_lgb = {
        "n_estimators": [80, 120],
        "learning_rate": [0.03, 0.05],
        "num_leaves": [20, 25],
        "max_depth": [5, 7]
    }

    grid_lgb = GridSearchCV(
        lgb.LGBMClassifier(random_state=42, verbose=-1),
        param_grid=param_grid_lgb,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid_lgb.fit(X_train_resampled, y_train_resampled)
    best_lgb = grid_lgb.best_estimator_

    y_pred_lgb = best_lgb.predict(X_test_scaled)
    y_proba_lgb = best_lgb.predict_proba(X_test_scaled)[:, 1]

    print(f"Best LGBM Params: {grid_lgb.best_params_}")
    print(f"LightGBM Test Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")
    print(classification_report(y_test, y_pred_lgb, target_names=["Not CKD", "CKD"]))

    plot_confusion_and_roc(y_test, y_pred_lgb, y_proba_lgb, "LightGBM")


if __name__ == "__main__":
    main()
