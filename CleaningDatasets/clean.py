import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\Asus\Desktop\AI_CourseWork\Datasets\oral_cancer_prediction_dataset.csv")

# Optional: preview columns
print("Original Columns:\n", df.columns.tolist())

# Define regression target
target = "Survival Rate (5-Year, %)"

# Drop unnecessary or classification-specific columns
drop_cols = [
    'ID',
    'Country',
    'Oral Cancer (Diagnosis)',
    'Early Diagnosis',
    'Economic Burden (Lost Workdays per Year)',
    # Optional drops depending on what you want to predict
    # 'Cost of Treatment (USD)',  # keep if you don't want to predict this
]

df = df.drop(columns=drop_cols)

# Keep only numeric features + categorical ones that may influence regression
# Identify categorical columns (object type)
cat_cols = df.select_dtypes(include='object').columns

# Encode categorical columns using LabelEncoder
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Optional: Remove rows with missing or zero target values
df = df[df[target] > 0]  # Remove rows where survival rate is 0

# Final dataset shape and preview
print("\nCleaned Dataset Shape:", df.shape)
print("\nSample Data:\n", df.head())
