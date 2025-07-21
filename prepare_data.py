# prepare_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class DataPreparation:
    def __init__(self, input_path):
        self.input_path = input_path
        self.df = None
        self.scaler = StandardScaler()

    def validate_data(self, df):
        """Validate the input data for required columns and valid values"""
        required_features = [
            'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
            'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
            'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
            'Compromised Immune System', 'Oral Cancer (Diagnosis)'
        ]
        
        # Check for required columns
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate binary columns contain only Yes/No
        binary_cols = [col for col in required_features if col != 'Diet (Fruits & Vegetables Intake)']
        for col in binary_cols:
            invalid_values = df[col].dropna().unique().tolist()
            invalid_values = [v for v in invalid_values if v not in ['Yes', 'No']]
            if invalid_values:
                raise ValueError(f"Column {col} contains invalid values: {invalid_values}")
        
        # Validate diet levels
        valid_diet_levels = ['Low', 'Moderate', 'High']
        invalid_diet = df['Diet (Fruits & Vegetables Intake)'].dropna().unique().tolist()
        invalid_diet = [v for v in invalid_diet if v not in valid_diet_levels]
        if invalid_diet:
            raise ValueError(f"Diet column contains invalid values: {invalid_diet}")
        
        return True

    def load_and_prepare(self):
        """Load, clean, encode, scale, and balance the dataset"""
        try:
            print(f"Loading data from: {self.input_path}")
            self.df = pd.read_csv(self.input_path)
            print("Original dataset:", self.df.shape)
            
            # Validate data before processing
            self.validate_data(self.df)
            
            # Calculate and print missing values
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                print("\nMissing values per column:")
                print(missing_values[missing_values > 0])
            
            # Relevant features for the model
            self.features = [
                'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
                'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
                'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
                'Compromised Immune System', 'Oral Cancer (Diagnosis)'
            ]
            
            # Select features and handle missing values
            self.df = self.df[self.features].copy()
            initial_size = len(self.df)
            self.df = self.df.dropna()
            dropped_rows = initial_size - len(self.df)
            if dropped_rows > 0:
                print(f"\nRemoved {dropped_rows} rows with missing values")
                print(f"Remaining dataset size: {len(self.df)}")
            
            print("\nSelected features:", self.features)

            # Map binary variables to 0/1
            binary_cols = [col for col in self.features if col != 'Diet (Fruits & Vegetables Intake)']
            
            # Print initial value distributions
            print("\nInitial value distributions:")
            for col in self.features:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
            
            # Encode binary variables
            for col in binary_cols:
                before_count = len(self.df)
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})
                after_count = len(self.df.dropna())
                if after_count < before_count:
                    print(f"\nWarning: {before_count - after_count} invalid values in {col}")

            # Convert and validate Diet levels
            diet_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            self.df['Diet (Fruits & Vegetables Intake)'] = self.df['Diet (Fruits & Vegetables Intake)'].map(diet_map)
            
            # Handle any remaining missing values
            missing_after_encoding = self.df.isnull().sum()
            if missing_after_encoding.any():
                print("\nMissing values after encoding:")
                print(missing_after_encoding[missing_after_encoding > 0])
                self.df = self.df.dropna()
                print(f"Removed {missing_after_encoding.sum()} rows with invalid values")

            # Scale the diet feature
            self.df['Diet (Fruits & Vegetables Intake)'] = self.scaler.fit_transform(
                self.df[['Diet (Fruits & Vegetables Intake)']]
            )

            # Shuffle the data
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Print final class distribution and dataset statistics
            print("\nFinal Dataset Statistics:")
            print("-" * 30)
            class_dist = self.df['Oral Cancer (Diagnosis)'].value_counts()
            print("\nClass distribution:")
            print(f"Negative cases (0): {class_dist[0]} ({class_dist[0]/len(self.df)*100:.1f}%)")
            print(f"Positive cases (1): {class_dist[1]} ({class_dist[1]/len(self.df)*100:.1f}%)")
            print(f"Total dataset size: {len(self.df)}")
            
            # Print feature statistics
            print("\nFeature Statistics:")
            print("-" * 30)
            for col in self.features[:-1]:  # Exclude target variable
                if col == 'Diet (Fruits & Vegetables Intake)':
                    print(f"\n{col} (scaled):")
                    print(self.df[col].describe())
                else:
                    pos_rate = self.df[col].mean() * 100
                    print(f"\n{col}:")
                    print(f"Positive rate: {pos_rate:.1f}%")
            
            return self.df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.input_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The dataset file is empty")
        except Exception as e:
            raise Exception(f"Error loading or validating data: {str(e)}")

    def save_prepared_data(self, output_path):
        df_prepared = self.load_and_prepare()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_prepared.to_csv(output_path, index=False)
        print(f"✅ Preprocessed data saved to: {output_path}")

def main():
    input_path = "Datasets/oral_cancer_prediction_dataset.csv"
    output_path = "Datasets/preprocessed_oral_cancer.csv"  # Removed 'balanced' since we're using full dataset

    dp = DataPreparation(input_path)
    dp.save_prepared_data(output_path)

if __name__ == "__main__":
    main()
