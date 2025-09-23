import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ckd_dataset1.csv')

# --- 1. Initial Inspection ---
print("--- Dataset Info ---")
df.info()

print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum().sort_values(ascending=False))
# --- 2. Data Cleaning ---

# Descriptive column names
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

# Identify numerical and categorical columns
numerical_cols = ['age', 'blood_pressure', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
categorical_cols = [col for col in df.columns if col not in numerical_cols]

# Correct data types and clean stray characters
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in categorical_cols:
    df[col] = df[col].str.strip()

# Impute missing values
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --- 3. Feature Encoding ---

# Correcting inconsistencies in categorical data
df['diabetes_mellitus'].replace(to_replace={'\tyes': 'yes'}, inplace=True)
df['coronary_artery_disease'].replace(to_replace={'\tno': 'no'}, inplace=True)
df['classification'].replace(to_replace={'ckd\t': 'ckd'}, inplace=True)

# Encoding categorical variables
df['red_blood_cells'] = df['red_blood_cells'].replace({'normal': 1, 'abnormal': 0})
df['pus_cell'] = df['pus_cell'].replace({'normal': 1, 'abnormal': 0})
df['pus_cell_clumps'] = df['pus_cell_clumps'].replace({'present': 1, 'notpresent': 0})
df['bacteria'] = df['bacteria'].replace({'present': 1, 'notpresent': 0})
df['hypertension'] = df['hypertension'].replace({'yes': 1, 'no': 0})
df['diabetes_mellitus'] = df['diabetes_mellitus'].replace({'yes': 1, 'no': 0})
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace({'yes': 1, 'no': 0})
df['appetite'] = df['appetite'].replace({'good': 1, 'poor': 0})
df['pedal_edema'] = df['pedal_edema'].replace({'yes': 1, 'no': 0})
df['anemia'] = df['anemia'].replace({'yes': 1, 'no': 0})
df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})

# One-hot encode nominal features
df = pd.get_dummies(df, columns=['specific_gravity', 'albumin', 'sugar'], drop_first=True)

print("\n--- Cleaned Dataset Info ---")
df.info()

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum().sum()) # Should be 0

# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.countplot(x='classification', data=df, palette='viridis')
plt.title('Distribution of Classes (1: CKD, 0: Not CKD)')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not CKD', 'CKD'])
plt.show()