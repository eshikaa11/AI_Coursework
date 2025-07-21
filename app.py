"""
Flask Application for Oral Cancer Survival Prediction
"""

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and feature names
models_dir = "results/models"

# Define the original feature names we want to show in the UI
original_features = [
    'Tobacco Use', 'Alcohol Consumption', 'HPV Infection',
    'Betel Quid Use', 'Chronic Sun Exposure', 'Poor Oral Hygiene',
    'Diet (Fruits & Vegetables Intake)', 'Family History of Cancer',
    'Compromised Immune System'
]

# Define feature descriptions
feature_descriptions = {
    'Tobacco Use': 'Does the patient use any form of tobacco products?',
    'Alcohol Consumption': 'Does the patient consume alcohol regularly?',
    'HPV Infection': 'Has the patient been diagnosed with HPV infection?',
    'Betel Quid Use': 'Does the patient use betel quid?',
    'Chronic Sun Exposure': 'Is there significant exposure to sun, especially on lips?',
    'Poor Oral Hygiene': 'Does the patient have poor oral hygiene?',
    'Diet (Fruits & Vegetables Intake)': 'What is the level of fruits and vegetables in patient\'s diet?',
    'Family History of Cancer': 'Is there any history of cancer in immediate family members?',
    'Compromised Immune System': 'Does the patient have a compromised immune system?'
}

try:
    # Load model and feature names
    model_path = os.path.join(models_dir, 'logistic_model.pkl')
    feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
    
    model = joblib.load(model_path)
    model_features = joblib.load(feature_names_path)
    print("✅ Model and features loaded successfully!")
    print(f"Model features: {model_features}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html', 
                         features=original_features,
                         descriptions=feature_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all feature values from the form
        feature_values = []
        for feature in original_features:
            if feature == 'Diet (Fruits & Vegetables Intake)':
                # Diet level is encoded as 0 (Low), 1 (Moderate), 2 (High)
                value = int(request.form.get(feature, 0))
            else:
                # All other features are binary
                value = int(request.form.get(feature) == '1')
            feature_values.append(value)
            
        # Create feature array matching the training data order
        features = feature_values
        
        print(f"Debug - Input features: {features}")  # Debug print
        
        # Make prediction with the model
        features_array = np.array(features).reshape(1, -1)
        prediction = bool(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][1])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability * 100  # Convert to percentage
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
