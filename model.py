from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model and the scaler
model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = float(request.form.get('age'))
        sex = request.form.get('sex')
        education = float(request.form.get('education'))
        employment_status = request.form.get('employment_status')
        dominant_hand = request.form.get('dominant_hand')
        visits = float(request.form.get('visits'))
        ses = float(request.form.get('ses'))

        # Additional fields for the remaining features
        physical_activity = request.form.get('physical_activity')
        smoking_status = request.form.get('smoking_status')
        dementia_family_history = request.form.get('dementia_family_history')
        depression_status = request.form.get('depression_status')
        sleep_quality = request.form.get('sleep_quality')

        # Map 'sex', 'dominant_hand', 'employment_status', 'smoking_status', 'dementia_family_history', 'depression_status', 'sleep_quality', and 'physical_activity' to numeric
        sex_encoded = 1 if sex == 'M' else 0  # 1 for Male, 0 for Female
        dominant_hand_encoded = 1 if dominant_hand == 'Right' else 0  # 1 for Right, 0 for Left
        employment_status_encoded = 1 if employment_status == 'Employed' else (0 if employment_status == 'Unemployed' else 2)  # 0 for Unemployed, 2 for Retired
        smoking_status_encoded = 1 if smoking_status == 'Smoker' else 0  # 1 for Smoker, 0 for Never Smoked
        dementia_family_history_encoded = 1 if dementia_family_history == 'Demented' else 0  # 1 for Demented, 0 for Non-Demented

        # Encode depression_status into numeric values
        if depression_status == "Not Depressed":
            depression_status_encoded = 0
        elif depression_status == "Mildly Depressed":
            depression_status_encoded = 1
        else:  # Depressed
            depression_status_encoded = 2

        # Encode sleep_quality into numeric values
        if sleep_quality == "8 hours":
            sleep_quality_encoded = 1
        elif sleep_quality == "Less than 8 hours":
            sleep_quality_encoded = 0
        else:  # More than 8 hours
            sleep_quality_encoded = 2

        # Encode physical_activity into numeric values
        if physical_activity == "None":
            physical_activity_encoded = 0
        elif physical_activity == "Sometimes":
            physical_activity_encoded = 1
        else:  # Daily
            physical_activity_encoded = 2

        # Prepare the feature array for prediction
        features = np.array([[age, sex_encoded, education, employment_status_encoded, dominant_hand_encoded, visits, ses,
                              physical_activity_encoded, smoking_status_encoded, dementia_family_history_encoded,
                              depression_status_encoded, sleep_quality_encoded]])

        # Scale the features using the same scaler from training
        features_scaled = scaler.transform(features)

        # Make prediction using the trained model
        prediction = model.predict(features_scaled)[0]

        # Render the result back to the page
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        # Handle any errors and display them on the page
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Get the port from the environment (Render sets it automatically)
    port = int(os.environ.get('PORT', 5000))  # Fallback to 5000 if no port is found
    app.run(host='0.0.0.0', port=port, debug=True)
