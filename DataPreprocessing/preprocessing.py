import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset (adjust path if needed)
df = pd.read_csv(r"C:\Users\Asus\Desktop\AI_CourseWork\Datasets\oral_cancer_prediction_dataset.csv")

# Drop irrelevant or classification-specific columns
drop_cols = [
    'ID', 'Country', 'Oral Cancer (Diagnosis)',
    'Early Diagnosis', 'Economic Burden (Lost Workdays per Year)'
]
df.drop(columns=drop_cols, inplace=True)

# Drop rows with 0 or missing target values
df = df[df["Survival Rate (5-Year, %)"] > 0]

# Encode categorical features
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Optional: fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Confirm preprocessing
print("✅ Preprocessed Data Sample:")
print(df.head())
