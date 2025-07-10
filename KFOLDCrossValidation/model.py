import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt



# 1. Load Dataset (change the path accordingly)
df = pd.read_csv(r"C:\Users\Asus\Desktop\AI_CourseWork\Datasets\oral_cancer_prediction_dataset.csv")

# 2. Preprocessing
drop_cols = [
    'ID', 'Country', 'Oral Cancer (Diagnosis)',
    'Early Diagnosis', 'Economic Burden (Lost Workdays per Year)'
]
df.drop(columns=drop_cols, inplace=True)

# Remove rows where Survival Rate is 0 or missing
df = df[df["Survival Rate (5-Year, %)"] > 0]

# Encode categorical columns
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# 3. Prepare features and target
target = "Survival Rate (5-Year, %)"
X = df.drop(columns=[target])
y = df[target]

# 4. Set up K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)

r2_scores = []
rmse_scores = []

print("Starting 5-Fold Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    r2 = r2_score(y_val, preds)
    mse = mean_squared_error(y_val, preds)
    rmse = sqrt(mse)

    
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}\n")

print(f"Average R² Score: {np.mean(r2_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")

# Optional: Plot Actual vs Predicted for last fold
plt.figure(figsize=(8,6))
plt.scatter(y_val, preds, alpha=0.4, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Survival Rate")
plt.ylabel("Predicted Survival Rate")
plt.title("Actual vs Predicted Survival Rate (Last Fold)")
plt.grid(True)
plt.show()
