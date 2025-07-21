# Oral Cancer Survival Prediction System
Course: Introduction to Artificial Intelligence (STW5000CEM)

## Project Overview
This project implements a machine learning solution to predict the 5-year survival rate of oral cancer patients. The system uses logistic regression and random forest classifiers to predict whether a patient's 5-year survival rate will be above or below 50%.

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA) with visualizations
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Web interface for predictions
- Cross-validation for robust evaluation

## Project Structure
```
AI_CourseWork/
├── Datasets/
│   ├── oral_cancer_prediction_dataset.csv    # Original dataset
│   ├── preprocessed_oral_cancer.csv          # Cleaned and preprocessed data
│   └── plots/                               # Visualization outputs
│       ├── class_distribution.png
│       ├── feature_distributions/
│       ├── correlations.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── feature_importance.png
├── models/                                  # Saved model files
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── feature_names.pkl
├── templates/                               # Web interface templates
│   └── index.html
├── prepare_data.py                         # Data preprocessing script
├── train_model.py                          # Model training and evaluation
├── app.py                                  # Flask web application
└── README.md                               # This file
```

## Requirements
- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  flask
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/eshikaa11/AI_Coursework.git
   cd AI_Coursework
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Data Preparation:
   ```bash
   python prepare_data.py
   ```
   This script will:
   - Load the raw dataset
   - Remove irrelevant columns
   - Handle missing values
   - Encode categorical variables
   - Scale numerical features
   - Save the preprocessed dataset

2. Model Training:
   ```bash
   python train_model.py
   ```
   This script will:
   - Perform EDA and generate visualizations
   - Train Logistic Regression and Random Forest models
   - Perform cross-validation and hyperparameter tuning
   - Generate performance metrics and plots
   - Save trained models

3. Web Interface:
   ```bash
   python app.py
   ```
   - Access the web interface at `http://localhost:5000`
   - Enter patient features
   - Get survival predictions from both models

## Model Performance
The system compares two models:
1. Logistic Regression
   - Accuracy: [Your accuracy score]
   - F1-Score: [Your F1 score]
   - ROC-AUC: [Your ROC-AUC score]

2. Random Forest
   - Accuracy: [Your accuracy score]
   - F1-Score: [Your F1 score]
   - ROC-AUC: [Your ROC-AUC score]

## Visualizations
- Class Distribution: Shows the balance between high and low survival rates
- Feature Distributions: Box plots showing feature distributions by survival class
- Correlation Matrix: Heatmap of feature correlations
- ROC Curves: Model performance comparison
- Feature Importance: Top predictive features for each model
- Confusion Matrices: Detailed breakdown of model predictions

## Citations
1. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

2. The pandas development team. (2023). *pandas-dev/pandas: Pandas* (Version [version number]). Zenodo. https://doi.org/10.5281/zenodo.3509134

3. Waskom, M. (2021). *seaborn: statistical data visualization*. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
[Your Name]
Student ID: [Your Student ID]
University: [Your University]

## Acknowledgments
- Dataset source: [Include the source of your oral cancer dataset]
- Course instructor: [Instructor's name]
- [Any other acknowledgments]
