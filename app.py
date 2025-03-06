from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React to communicate with Flask

# Load the trained Random Forest model
rf_model = joblib.load('eye_strain_rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)  # Convert to 2D array
        
        # Make prediction
        prediction = rf_model.predict(features)

        # Return prediction
        return jsonify({'strain_level': int(prediction[0])})  # Ensure JSON compatibility
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # Use Render-assigned port or default to 10000
    app.run(host="0.0.0.0", port=port)

