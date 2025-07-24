"""
Machine Learning Model Training for Oral Cancer Risk Assessment
===============================================================

This script handles the complete machine learning pipeline for oral cancer
risk prediction, including:
- Data loading and validation
- Feature preprocessing and engineering
- Model training with multiple algorithms
- Cross-validation and hyperparameter tuning
- Model evaluation and performance metrics
- Model persistence for web application use

The trained models are saved to results/models/ directory for use by the
Flask web application.

Author: AI CourseWork Project
Course: Introduction to Artificial Intelligence (STW5000CEM)
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
# =============
DATA_PATH = "Datasets/preprocessed_oral_cancer.csv"
MODELS_DIR = "results/models"
PLOTS_DIR = "results/plots"
RANDOM_STATE = 42

# Feature definitions
FEATURE_COLUMNS = [
    'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
    'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
    'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
    'Compromised Immune System'
]

TARGET_COLUMN = 'Oral Cancer (Diagnosis)'

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

    def perform_kfold_validation(self, X, y, k=10):
        """Perform k-fold cross-validation and return detailed metrics"""
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Initialize metric lists
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        print(f"\nPerforming {k}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            # Split data
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Apply SMOTE to training fold only
            smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
            X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train, y_fold_train)
            
            # Train model on the fold
            model = LogisticRegression(**self.best_params, max_iter=2000)
            model.fit(X_fold_train_resampled, y_fold_train_resampled)
            
            # Make predictions
            y_pred = model.predict(X_fold_val)
            
            # Calculate metrics
            accuracies.append(accuracy_score(y_fold_val, y_pred))
            precisions.append(precision_score(y_fold_val, y_pred))
            recalls.append(recall_score(y_fold_val, y_pred))
            f1_scores.append(f1_score(y_fold_val, y_pred))
            
            print(f"Fold {fold}/{k} completed")
        
        # Calculate and print average metrics
        print("\nK-fold Cross-validation Results:")
        print(f"Average Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")
        print(f"Average Precision: {np.mean(precisions):.3f} (±{np.std(precisions):.3f})")
        print(f"Average Recall: {np.mean(recalls):.3f} (±{np.std(recalls):.3f})")
        print(f"Average F1-Score: {np.mean(f1_scores):.3f} (±{np.std(f1_scores):.3f})")
        
        return {
            'accuracy': (np.mean(accuracies), np.std(accuracies)),
            'precision': (np.mean(precisions), np.std(precisions)),
            'recall': (np.mean(recalls), np.std(recalls)),
            'f1': (np.mean(f1_scores), np.std(f1_scores))
        }

    def train_logistic_model(self):
        # Adjust hyperparameter grid for better handling of imbalanced data
        grid = GridSearchCV(
            LogisticRegression(max_iter=2000),
            param_grid={
                'C': [0.001, 0.01, 0.1, 1.0, 10.0],  # Extended C range
                'class_weight': ['balanced'],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'newton-cg'],  # Added another solver
            },
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            refit='f1',  # Use F1 score for best model selection
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Train the model with grid search
        print("\nPerforming grid search with cross-validation...")
        grid.fit(self.X_train, self.y_train)
        
        # Store best parameters
        self.best_params = grid.best_params_
        print("\nBest parameters:", self.best_params)
        print("Best F1 score:", grid.best_score_)
        
        # Perform detailed k-fold validation
        X_array = np.array(self.X_train)
        y_array = np.array(self.y_train)
        self.kfold_metrics = self.perform_kfold_validation(X_array, y_array)
        
        # Train final model with best parameters on full training set
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

        print("\n=== Model Evaluation ===")
        print("\n1. Classification Report on Test Set:")
        print(classification_report(self.y_test, y_pred, zero_division=1))
        
        print("\n2. K-fold Cross-validation Metrics:")
        print("Accuracy: {:.3f} (±{:.3f})".format(*self.kfold_metrics['accuracy']))
        print("Precision: {:.3f} (±{:.3f})".format(*self.kfold_metrics['precision']))
        print("Recall: {:.3f} (±{:.3f})".format(*self.kfold_metrics['recall']))
        print("F1-Score: {:.3f} (±{:.3f})".format(*self.kfold_metrics['f1']))

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
