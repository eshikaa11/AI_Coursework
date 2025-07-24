"""
Oral Cancer Risk Assessment - Data Preprocessing Module
=====================================================

This module handles data preprocessing for the oral cancer prediction model.
It includes data validation, encoding, scaling, and preparation for machine learning.

Authors: AI Assistant
Date: 2024
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OralCancerDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for oral cancer risk assessment.
    
    Features:
    - Data validation and quality checks
    - Missing value handling
    - Feature encoding (binary and ordinal)
    - Data scaling and normalization
    - Exploratory data analysis
    - Clean data export
    
    Attributes:
        input_path (str): Path to raw dataset
        df (pd.DataFrame): Processed dataset
        scaler (StandardScaler): Feature scaler for numerical features
    """
    
    def __init__(self, input_path):
        """
        Initialize the data preprocessor.
        
        Args:
            input_path (str): Path to the raw CSV dataset
        """
        self.input_path = input_path
        self.df = None
        self.scaler = StandardScaler()
        
        # Feature configurations
        self.required_features = [
            'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
            'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
            'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
            'Compromised Immune System', 'Oral Cancer (Diagnosis)'
        ]
        
        self.binary_features = [
            'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
            'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
            'Family History of Cancer', 'Compromised Immune System',
            'Oral Cancer (Diagnosis)'
        ]
        
        self.ordinal_features = ['Diet (Fruits & Vegetables Intake)']
        
        print("🔄 Oral Cancer Data Preprocessor Initialized")
        print(f"📁 Input dataset: {input_path}")

    def validate_data(self, df):
        """
        Comprehensive data validation checks.
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        print("\n🔍 Validating dataset...")
        
        # Check for required columns
        missing_cols = [col for col in self.required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Validate binary columns
        for col in self.binary_features:
            unique_vals = df[col].dropna().unique()
            invalid_vals = [v for v in unique_vals if v not in ['Yes', 'No', 1, 0]]
            if invalid_vals:
                raise ValueError(f"❌ Column '{col}' contains invalid values: {invalid_vals}")
        
        # Validate ordinal features
        valid_diet_levels = ['Low', 'Moderate', 'High']
        diet_vals = df['Diet (Fruits & Vegetables Intake)'].dropna().unique()
        invalid_diet = [v for v in diet_vals if v not in valid_diet_levels]
        if invalid_diet:
            raise ValueError(f"❌ Diet column contains invalid values: {invalid_diet}")
        
        print("✅ Data validation passed")
        return True

    def load_and_prepare(self):
        """
        Complete data preprocessing pipeline.
        
        Steps:
        1. Load raw data
        2. Validate data quality
        3. Handle missing values
        4. Encode categorical features
        5. Scale numerical features
        6. Generate data statistics
        
        Returns:
            pd.DataFrame: Processed and cleaned dataset
        """
        print("\n📊 Loading and preparing dataset...")
        
        try:
            # Load raw data
            self.df = pd.read_csv(self.input_path)
            print(f"✅ Loaded dataset: {self.df.shape}")
            
            # Validate data structure
            self.validate_data(self.df)
            
            # Filter to required features only
            self.df = self.df[self.required_features].copy()
            
            # Handle missing values
            initial_size = len(self.df)
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print(f"\n⚠️ Missing values found:")
                for col, count in missing_values[missing_values > 0].items():
                    print(f"   {col}: {count} missing")
                
                # Drop rows with missing values
                self.df = self.df.dropna()
                dropped = initial_size - len(self.df)
                print(f"🧹 Removed {dropped} rows with missing values")
            
            print(f"📈 Dataset after cleaning: {self.df.shape}")
            
            # Show original value distributions
            print("\n📋 Original value distributions:")
            for col in self.required_features:
                if col != 'Oral Cancer (Diagnosis)':  # Skip target for now
                    print(f"\n{col}:")
                    print(self.df[col].value_counts().to_dict())
            
            # Encode binary features (Yes/No → 1/0)
            print("\n🔄 Encoding binary features...")
            for col in self.binary_features:
                if col in self.df.columns:
                    self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})
                    print(f"   ✅ {col}: Yes/No → 1/0")
            
            # Encode ordinal features (Diet levels)
            print("\n🔄 Encoding ordinal features...")
            diet_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
            self.df['Diet (Fruits & Vegetables Intake)'] = self.df['Diet (Fruits & Vegetables Intake)'].map(diet_mapping)
            print("   ✅ Diet: Low/Moderate/High → 0/1/2")
            
            # Scale diet feature
            diet_scaled = self.scaler.fit_transform(self.df[['Diet (Fruits & Vegetables Intake)']])
            self.df['Diet (Fruits & Vegetables Intake)'] = diet_scaled.flatten()
            print("   ✅ Diet feature scaled")
            
            # Shuffle dataset
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            print("🔀 Dataset shuffled")
            
            # Final statistics
            self._print_final_statistics()
            
            return self.df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Dataset not found: {self.input_path}")
        except Exception as e:
            raise Exception(f"❌ Preprocessing failed: {str(e)}")
    
    def _print_final_statistics(self):
        """Print comprehensive dataset statistics."""
        print("\n" + "="*50)
        print("📊 FINAL DATASET STATISTICS")
        print("="*50)
        
        # Class distribution
        target_dist = self.df['Oral Cancer (Diagnosis)'].value_counts()
        total = len(self.df)
        
        print(f"\n🎯 Target Variable Distribution:")
        print(f"   No Cancer (0): {target_dist[0]:,} ({target_dist[0]/total*100:.1f}%)")
        print(f"   Cancer (1):    {target_dist[1]:,} ({target_dist[1]/total*100:.1f}%)")
        print(f"   Total samples: {total:,}")
        
        # Feature statistics
        print(f"\n📈 Feature Statistics:")
        for col in self.required_features[:-1]:  # Exclude target
            if col == 'Diet (Fruits & Vegetables Intake)':
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                print(f"   {col} (scaled): μ={mean_val:.3f}, σ={std_val:.3f}")
            else:
                positive_rate = self.df[col].mean() * 100
                print(f"   {col}: {positive_rate:.1f}% positive")
        
        print("="*50)

    def save_prepared_data(self, output_path):
        """
        Execute preprocessing pipeline and save cleaned data.
        
        Args:
            output_path (str): Path to save the processed dataset
        """
        print(f"\n💾 Saving processed data to: {output_path}")
        
        # Execute preprocessing
        processed_df = self.load_and_prepare()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed dataset
        processed_df.to_csv(output_path, index=False)
        
        print(f"✅ Processed dataset saved successfully!")
        print(f"📁 Location: {output_path}")
        print(f"📊 Final shape: {processed_df.shape}")
        
        return processed_df

def main():
    """
    Main preprocessing pipeline execution.
    Processes raw data and saves cleaned dataset.
    """
    print("🦷 Oral Cancer Risk Assessment - Data Preprocessing")
    print("=" * 60)
    
    # Configuration
    input_path = "Datasets/oral_cancer_prediction_dataset.csv"
    output_path = "Datasets/preprocessed_oral_cancer.csv"
    
    try:
        # Initialize preprocessor
        preprocessor = OralCancerDataPreprocessor(input_path)
        
        # Execute preprocessing pipeline
        preprocessor.save_prepared_data(output_path)
        
        print("\n" + "=" * 60)
        print("🎉 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Output file: {output_path}")
        print("\n🚀 Next step: Train the model:")
        print("   python train_model.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """Script entry point"""
    main()
