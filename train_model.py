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
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for documentation
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300

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
        self.scaler = StandardScaler()
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare_features(self):
        X = self.df.drop('Oral Cancer (Diagnosis)', axis=1)
        y = self.df['Oral Cancer (Diagnosis)']

        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("\nOriginal dataset shapes:")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print("\nOriginal class distribution in training set:")
        print(pd.Series(y_train).value_counts())

        # Apply feature scaling
        print("\n🔄 Applying feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("✅ Feature scaling completed")

        # Apply SMOTE for class balancing (after scaling)
        print("\n⚖️ Applying SMOTE for class balancing...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print("✅ SMOTE balancing completed")
        except ImportError:
            print("⚠️ SMOTE not available, using alternative resampling...")
            # Fallback to original resampling method if SMOTE not available
            df_train = pd.DataFrame(X_train_scaled, columns=X.columns)
            df_train['target'] = y_train.values
            
            df_majority = df_train[df_train.target == 0]
            df_minority = df_train[df_train.target == 1]
            
            df_minority_upsampled = resample(df_minority, 
                                           replace=True,
                                           n_samples=len(df_majority),
                                           random_state=42)
            
            df_resampled = pd.concat([df_majority, df_minority_upsampled])
            X_train_resampled = df_resampled.drop('target', axis=1).values
            y_train_resampled = df_resampled.target.values

        # Store the processed data
        self.X_train = X_train_resampled
        self.y_train = y_train_resampled
        self.X_test = X_test_scaled
        self.y_test = y_test

        # Print final dataset shapes and class distributions
        print("\nFinal dataset shapes:")
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        print("\nFinal class distribution in training set:")
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
            
            # Scale features for current fold
            fold_scaler = StandardScaler()
            X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = fold_scaler.transform(X_fold_val)
            
            # Handle class imbalance using SMOTE or fallback method
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_scaled, y_fold_train)
            except ImportError:
                # Fallback to resample method
                df_fold = pd.DataFrame(X_fold_train_scaled)
                df_fold['target'] = y_fold_train
                df_majority = df_fold[df_fold.target == 0]
                df_minority = df_fold[df_fold.target == 1]
                
                if len(df_minority) > 0 and len(df_majority) > 0:
                    df_minority_upsampled = resample(df_minority, 
                                                   replace=True,
                                                   n_samples=len(df_majority),
                                                   random_state=42)
                    df_resampled = pd.concat([df_majority, df_minority_upsampled])
                    X_fold_train_resampled = df_resampled.drop('target', axis=1).values
                    y_fold_train_resampled = df_resampled.target.values
                else:
                    X_fold_train_resampled = X_fold_train_scaled
                    y_fold_train_resampled = y_fold_train
            
            # Train model on the fold
            model = LogisticRegression(**self.best_params, max_iter=2000)
            model.fit(X_fold_train_resampled, y_fold_train_resampled)
            
            # Make predictions
            y_pred = model.predict(X_fold_val_scaled)
            
            # Calculate metrics
            accuracies.append(accuracy_score(y_fold_val, y_pred))
            precisions.append(precision_score(y_fold_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_fold_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_fold_val, y_pred, zero_division=0))
            
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
        
        # Perform detailed k-fold validation on original data (before resampling)
        original_X = self.df.drop('Oral Cancer (Diagnosis)', axis=1).values
        original_y = self.df['Oral Cancer (Diagnosis)'].values
        self.kfold_metrics = self.perform_kfold_validation(original_X, original_y)
        
        # Train final model with best parameters on full training set
        self.model = grid.best_estimator_
        
        # Feature importance analysis (need to use column names from original data)
        feature_names = self.df.drop('Oral Cancer (Diagnosis)', axis=1).columns
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(self.model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Create feature importance visualization
        self.create_feature_importance_plot(feature_importance)
        
        # Create class distribution visualization
        self.create_class_distribution_plot()

    def create_feature_importance_plot(self, feature_importance, save_dir="results/plots"):
        """Create a simple feature importance visualization for documentation."""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
                       color='skyblue', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance - Oral Cancer Risk Factors', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{save_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved to: {save_dir}/feature_importance.png")

    def create_class_distribution_plot(self, save_dir="results/plots"):
        """Create a simple class distribution visualization for documentation."""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Class Distribution: Before and After Balancing', fontsize=14, fontweight='bold')
        
        # Original distribution (from the full dataset)
        original_dist = self.df['Oral Cancer (Diagnosis)'].value_counts()
        colors = ['lightgreen', 'lightcoral']
        
        ax1.pie(original_dist.values, labels=['No Cancer', 'Cancer'], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Original Dataset', fontweight='bold')
        
        # Training set distribution (after balancing)
        train_dist = pd.Series(self.y_train).value_counts()
        ax2.pie(train_dist.values, labels=['No Cancer', 'Cancer'], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Training Set (After Balancing)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Class distribution plot saved to: {save_dir}/class_distribution.png")

    def create_model_performance_summary(self, save_dir="results/plots"):
        """Create a simple model performance summary visualization."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions for visualization
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Create a 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['No Cancer', 'Cancer'], 
                   yticklabels=['No Cancer', 'Cancer'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)
        
        # 3. Performance Metrics Bar Chart
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred)
        }
        
        bars = ax3.bar(metrics.keys(), metrics.values(), 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'],
                      alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics', fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Class Distribution in Test Set
        test_dist = pd.Series(self.y_test).value_counts()
        colors = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax4.pie(test_dist.values, 
                                          labels=['No Cancer', 'Cancer'], 
                                          autopct='%1.1f%%', 
                                          startangle=90,
                                          colors=colors)
        ax4.set_title('Test Set Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plt.savefig(f"{save_dir}/model_performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model performance summary saved to: {save_dir}/model_performance_summary.png")

    def evaluate_model(self, save_dir="results/plots"):
        os.makedirs(save_dir, exist_ok=True)
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print("\n" + "="*60)
        print("🔍 MODEL EVALUATION RESULTS")
        print("="*60)
        
        print("\n1. Test Set Performance:")
        print(f"   Accuracy:  {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"   Precision: {precision_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"   Recall:    {recall_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"   F1-Score:  {f1_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"   AUC-ROC:   {roc_auc_score(self.y_test, y_prob):.3f}")

        print("\n2. Classification Report on Test Set:")
        print(classification_report(self.y_test, y_pred, zero_division=1))
        
        print("\n3. K-fold Cross-validation Metrics:")
        print("   Accuracy:  {:.3f} (±{:.3f})".format(*self.kfold_metrics['accuracy']))
        print("   Precision: {:.3f} (±{:.3f})".format(*self.kfold_metrics['precision']))
        print("   Recall:    {:.3f} (±{:.3f})".format(*self.kfold_metrics['recall']))
        print("   F1-Score:  {:.3f} (±{:.3f})".format(*self.kfold_metrics['f1']))

        # Create comprehensive model performance visualization
        self.create_model_performance_summary(save_dir)

        # Individual plots (keeping the original simple ones)
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
        
        print(f"\n📊 All visualizations saved to: {save_dir}")
        print("="*60)

    def save_model(self, save_dir="results/models"):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, f"{save_dir}/logistic_model.pkl")
        joblib.dump(self.scaler, f"{save_dir}/scaler.pkl")
        
        # Save feature names from original dataframe
        feature_names = self.df.drop('Oral Cancer (Diagnosis)', axis=1).columns.tolist()
        joblib.dump(feature_names, f"{save_dir}/feature_names.pkl")
        
        print(f"✅ Model saved to: {save_dir}")
        print(f"✅ Scaler saved to: {save_dir}")
        print(f"✅ Feature names saved to: {save_dir}")

def main():
    dataset_path = "Datasets/preprocessed_oral_cancer.csv"
    model = OralCancerModel(dataset_path)
    model.prepare_features()
    model.train_logistic_model()
    model.evaluate_model()
    model.save_model()

if __name__ == "__main__":
    main()
