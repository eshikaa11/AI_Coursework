# Oral Cancer Risk Assessment System
Course: Introduction to Artificial Intelligence (STW5000CEM)

## Project Overview
This project implements a machine learning-based risk assessment system for oral cancer. Using a comprehensive set of risk factors, the system evaluates an individual's risk profile and provides personalized recommendations. The model utilizes logistic regression with advanced preprocessing techniques and balanced class weights to ensure reliable risk predictions.

## Key Features
- **Risk Factor Analysis**: Evaluates 9 critical risk factors:
  - Tobacco Use
  - Alcohol Consumption
  - HPV Infection
  - Betel Quid Use
  - Chronic Sun Exposure
  - Poor Oral Hygiene
  - Diet (Fruits & Vegetables Intake)
  - Family History of Cancer
  - Compromised Immune System

- **Advanced Data Processing**:
  - Comprehensive data validation
  - Intelligent handling of missing values
  - Feature scaling for diet levels
  - Preservation of natural class distributions

- **Interactive Web Interface**:
  - User-friendly form with organized sections
  - Real-time validation
  - Dynamic risk assessment visualization
  - Personalized recommendations
  - Detailed risk factor breakdown

- **Machine Learning Implementation**:
  - Logistic Regression with balanced class weights
  - Cross-validation for robust model evaluation
  - Hyperparameter optimization
  - Feature importance analysis

## Project Structure
```
AI_CourseWork/
├── Datasets/
│   ├── oral_cancer_prediction_dataset.csv    # Original dataset
│   ├── preprocessed_oral_cancer.csv          # Processed dataset
│   └── plots/                               # Visualization outputs
│       ├── confusion_matrix.png
│       └── roc_curve.png
├── results/
│   └── models/                             # Saved model files
│       ├── model.pkl                       # Trained model
│       └── feature_names.pkl               # Feature order
├── templates/                              # Web interface templates
│   └── index.html                         # Interactive assessment form
├── prepare_data.py                        # Data preprocessing script
├── train_model.py                         # Model training script
├── app.py                                 # Flask web application
└── README.md                              # Documentation
```

## Technical Requirements
- Python 3.8 or higher
- Required packages:
  ```
  pandas>=1.5.0
  numpy>=1.20.0
  scikit-learn>=1.0.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  flask>=2.0.0
  joblib>=1.1.0
  imblearn>=0.10.0
  ```

## Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/eshikaa11/AI_Coursework.git
   cd AI_Coursework
   ```

2. Set up a Python virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # For Windows:
   .\venv\Scripts\activate
   # For Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn flask joblib imblearn
   ```

## Usage Guide

1. **Data Preprocessing**:
   ```bash
   python prepare_data.py
   ```
   This script performs:
   - Data validation and cleaning
   - Feature encoding (binary and categorical)
   - Diet level scaling
   - Missing value handling
   - Data quality reporting

2. **Model Training**:
   ```bash
   python train_model.py
   ```
   This process includes:
   - Feature preparation
   - SMOTE for handling imbalanced data
   - Grid search for hyperparameter optimization
   - Model evaluation with metrics
   - Feature importance analysis
   - Model persistence

3. **Launch Web Interface**:
   ```bash
   python app.py
   ```
   Access the application:
   - Open browser to `http://localhost:5000`
   - Fill in the risk assessment form
   - View real-time risk evaluation
   - Get personalized recommendations

4. **Using the Risk Assessment Tool**:
   - Complete all sections of the form:
     * Substance Use History
     * Medical Conditions
     * Lifestyle & Environmental Factors
     * Family History
   - Review the comprehensive risk assessment
   - Check personalized recommendations
   - Note the risk level and contributing factors

## System Features

### Data Processing Capabilities
- Comprehensive data validation
- Intelligent missing value handling
- Multi-level feature encoding
- Advanced diet level scaling
- Data quality reporting

### Model Implementation
- Logistic Regression with balanced weights
- SMOTE for handling class imbalance
- Grid search optimization
- Cross-validation for robustness
- Feature importance analysis

### Web Interface Features
- Organized risk factor sections
- Real-time form validation
- Dynamic risk visualization
- Personalized recommendations
- Detailed risk factor breakdown
- Mobile-responsive design

### Visualization Outputs
- ROC curve analysis
- Confusion matrix
- Feature importance plots
- Risk level indicators
- Interactive progress bars

## Implementation Details

### Risk Assessment Components
1. **Primary Risk Factors**:
   - Substance use (tobacco, alcohol, betel quid)
   - Medical conditions (HPV, immune system)
   - Lifestyle factors (diet, hygiene, sun exposure)
   - Genetic factors (family history)

2. **Risk Level Categories**:
   - Low Risk (< 25%)
   - Moderate Risk (25-50%)
   - High Risk (50-75%)
   - Very High Risk (> 75%)

3. **Recommendation System**:
   - Personalized health advice
   - Lifestyle modification suggestions
   - Medical consultation guidance
   - Preventive measures

### Technical Implementation
1. **Frontend**:
   - Bootstrap 5 components
   - jQuery for AJAX calls
   - Dynamic form validation
   - Interactive visualizations

2. **Backend**:
   - Flask REST API
   - Scikit-learn integration
   - Secure data handling
   - Error management

3. **Model Pipeline**:
   - Data validation
   - Feature preprocessing
   - Risk calculation
   - Results formatting

## References

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

2. Lemaître, G., et al. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets. *Journal of Machine Learning Research, 18*(17), 1-5.

## Author
Student Name: [Your Name]
Student ID: [Your Student ID]
Course: Introduction to Artificial Intelligence (STW5000CEM)
University: [Your University Name]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset Provider: [Source of the oral cancer dataset]
- Academic Supervisor: [Instructor's Name]
- Faculty of [Your Department]
