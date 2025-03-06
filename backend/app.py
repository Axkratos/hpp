from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
import os
import joblib
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import pickle
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Initialize CORS for universal access
CORS(app, resources={r"/api/*": {"origins": "*"}},
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"])

# Set up logging
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('App startup')

# Load the pre-trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the features to be scaled
numeric_cols = ['Bedroom', 'Bathroom', 'Floors', 'Year', 'Road Width', 'Area_in_sqft']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        feature_names = data.get('feature_names')  # Expect feature names in request
        features = data.get('features')
        
        if not feature_names:
            raise ValueError("Feature names are required")

        if len(features) != len(feature_names):  # Check if features length matches feature names
            raise ValueError(f"Expected {len(feature_names)} features, got {len(features)}")

        app.logger.info("Received features: %s", features)
        app.logger.info("Feature names: %s", feature_names)

        # Convert features to a NumPy array
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)

        # Extract the relevant features for scaling
        features_dict = {name: value for name, value in zip(feature_names, features)}
        features_to_scale = [features_dict.get(col, 0) for col in numeric_cols]  # Default to 0 if not found
        
        # Convert to NumPy array and reshape
        features_to_scale_array = np.array(features_to_scale, dtype=np.float64).reshape(1, -1)
        app.logger.info("Features to scale: %s", features_to_scale)

        # Scale features
        features_scaled = scaler.transform(features_to_scale_array)
        app.logger.info("Scaled features: %s", features_scaled)

        # Prepare the final feature array with scaled features
        final_features = features_array.copy()
        for i, col in enumerate(numeric_cols):
            feature_index = feature_names.index(col)
            final_features[0, feature_index] = features_scaled[0, i]
        
        app.logger.info("Final features for prediction: %s", final_features)

        # Make prediction
        prediction = model.predict(final_features)
        prediction_value = float(prediction[0])
        app.logger.info("Prediction: %f", prediction_value)

        # Return the prediction
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error("Error in prediction: %s", e)
        return jsonify({'error': str(e)}), 400

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = '#####@gmail.com'
app.config['MAIL_PASSWORD'] = '####'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_DEFAULT_SENDER'] = '#####@gmail.com'

mail = Mail(app)

# Define a route for the contact form submission
@app.route('/api/contact', methods=['POST'])
def contact():
    try:
        # Extract data from the request
        full_name = request.form.get('fullName')
        email = request.form.get('email')
        message_body = request.form.get('messageBody')

        app.logger.info("Contact form data - Name: %s, Email: %s, Message: %s", full_name, email, message_body)

        # Validate the input
        if not full_name or not email or not message_body:
            app.logger.warning("Validation error: Missing fields")
            return jsonify({'error': 'All fields are required'}), 400

        # Create the email message
        msg = Message(subject='Contact Us Form Submission',
                      recipients=[app.config['MAIL_DEFAULT_SENDER']],
                      body=f"Name: {full_name}\nEmail: {email}\nMessage: {message_body}")

        # Send the email
        mail.send(msg)
        app.logger.info("Email sent successfully")

        return jsonify({'message': 'Message sent successfully'}), 200

    except Exception as e:
        app.logger.error("Error in contact form submission: %s", e)
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
