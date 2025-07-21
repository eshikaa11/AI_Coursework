# train_model.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import joblib

class OralCancerModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare_features(self):
        X = self.df.drop('Oral Cancer (Diagnosis)', axis=1)
        y = self.df['Oral Cancer (Diagnosis)']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply SMOTE with adjusted sampling strategy
        smote = SMOTE(
            random_state=42,
            sampling_strategy='auto',  # automatically determine the sampling ratio
            k_neighbors=5  # reduce neighbors for better boundary handling
        )
        self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        self.X_test, self.y_test = X_test, y_test

        # Print dataset shapes and class distributions
        print("\nDataset shapes:")
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print("\nClass distribution in training set:")
        print(pd.Series(self.y_train).value_counts())
        print("\nClass distribution in test set:")
        print(pd.Series(self.y_test).value_counts())

    def train_logistic_model(self):
        # Adjust hyperparameter grid for better handling of imbalanced data
        grid = GridSearchCV(
            LogisticRegression(max_iter=2000),  # Increase max iterations
            param_grid={
                'C': [0.01, 0.1, 1.0, 10.0],  # Simplified C range
                'class_weight': ['balanced'],  # Always use balanced class weights
                'penalty': ['l2'],  # Use L2 regularization for stability
                'solver': ['lbfgs'],  # Use robust solver
            },
            scoring='balanced_accuracy',  # Use balanced accuracy for imbalanced data
            cv=5,  # Reduce cross-validation folds for stability
            n_jobs=-1,
            verbose=1
        )
        
        # Train the model
        print("\nTraining model with cross-validation...")
        grid.fit(self.X_train, self.y_train)
        
        # Print results
        print("\nBest parameters:", grid.best_params_)
        print("Best ROC-AUC score:", grid.best_score_)
        self.model = grid.best_estimator_
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': abs(self.model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)

    def evaluate_model(self, save_dir="results/plots"):
        os.makedirs(save_dir, exist_ok=True)
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=1))  # Handle zero division cases

        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(f"{save_dir}/roc_curve.png")
        plt.close()

    def save_model(self, save_dir="results/models"):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, f"{save_dir}/logistic_model.pkl")
        joblib.dump(self.X_train.columns.tolist(), f"{save_dir}/feature_names.pkl")
        print(f"✅ Model saved to: {save_dir}")

def main():
    dataset_path = "Datasets/preprocessed_oral_cancer_balanced.csv"
    model = OralCancerModel(dataset_path)
    model.prepare_features()
    model.train_logistic_model()
    model.evaluate_model()
    model.save_model()

if __name__ == "__main__":
    main()
