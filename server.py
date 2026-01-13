import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin requests

# Configuration
MODEL_FOLDER = 'models'
TARGET_PHRASE = "the quick brown fox jumps over the lazy dog"

# Ensure model folder exists
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

def extract_features(keystroke_data):
    """
    FIXED: Extract features to match the 41-feature model.
    Hypothesis: The model uses Flight Times only, excluding the first index.
    """
    flight_times = []
    
    for i in range(1, len(keystroke_data)):
        # Flight time: Time between releasing key N and pressing key N+1
        curr_press = keystroke_data[i]['pressTime']
        prev_release = keystroke_data[i-1]['releaseTime']
        flight_times.append(curr_press - prev_release)
            
    # We expect 42 flight times for a 43-character phrase.
    # Your model expects 41 features.
    # We will take the LAST 41 flight times (dropping the first one).
    
    if len(flight_times) > 41:
        flight_times = flight_times[-41:]
    
    # Safety pad if typing was too short (prevents crash)
    while len(flight_times) < 41:
        flight_times.append(0)
    
    return np.array(flight_times).reshape(1, -1)

@app.route('/')
def index():
    # Serve the HTML file if accessed directly
    return send_from_directory('.', 'verification.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    """Scans the 'models' folder and returns available user IDs"""
    users = []
    try:
        for filename in os.listdir(MODEL_FOLDER):
            if filename.endswith('.pkl'):
                # Removing .pkl extension to get user ID
                user_id = filename[:-4]
                users.append(user_id)
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/verify', methods=['POST'])
def verify_user():
    data = request.json
    claimed_user = data.get('claimedUser')
    keystrokes = data.get('keystrokes')
    
    if not claimed_user or not keystrokes:
        return jsonify({"authenticated": False, "message": "Missing data"}), 400

    model_path = os.path.join(MODEL_FOLDER, f"{claimed_user}.pkl")
    
    if not os.path.exists(model_path):
        return jsonify({"authenticated": False, "message": "User model not found"}), 404

    try:
        loaded_object = joblib.load(model_path)
        
        clf = None
        scaler = None

        # 1. Unpack the Dictionary
        if isinstance(loaded_object, dict):
            # Debug: Print the expected columns to console so you can see them
            if 'feature_columns' in loaded_object:
                cols = loaded_object['feature_columns']
                print(f"DEBUG: Model expects {len(cols)} columns: {cols[:3]}...")
            
            clf = loaded_object.get('model') or loaded_object.get('clf')
            scaler = loaded_object.get('scaler')
        else:
            clf = loaded_object

        if clf is None:
            return jsonify({"authenticated": False, "message": "Invalid model file structure"}), 500

        # 2. Extract Features
        feature_vector = extract_features(keystrokes)
        
        # 3. Scale (CRITICAL)
        # We wrap this in a try-catch specifically for shape mismatch
        try:
            if scaler:
                feature_vector = scaler.transform(feature_vector)
        except ValueError as ve:
            # If scaling fails, it means our extraction is still wrong.
            return jsonify({
                "authenticated": False, 
                "message": f"Shape Mismatch: Model wants {scaler.n_features_in_} features, you sent {feature_vector.shape[1]}."
            }), 500

        # 4. Predict
        prediction = clf.predict(feature_vector)
        # Isolation Forest: 1 = Inlier (Auth), -1 = Outlier (Intruder)
        is_authenticated = prediction[0] == 1
        
        # Calculate Score
        score = clf.decision_function(feature_vector)[0]
        
        # Visual Confidence Score
        confidence = 0
        if is_authenticated:
            # Normalize score (usually roughly -0.5 to 0.5) to a %
            confidence = min(max((score + 0.2) * 200, 50), 98)

        return jsonify({
            "authenticated": bool(is_authenticated),
            "confidence": round(confidence, 1),
            "user": claimed_user,
            "score": float(score)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"authenticated": False, "message": f"Server Error: {str(e)}"}), 500
    except Exception as e:
        # This prints the exact error to your terminal so you can see what's wrong
        import traceback
        traceback.print_exc() 
        return jsonify({"authenticated": False, "message": f"Server Error: {str(e)}"}), 500
    
if __name__ == '__main__':
    print(f"Starting server... Place your .pkl files in the '{MODEL_FOLDER}' folder.")
    app.run(debug=True, port=5000)