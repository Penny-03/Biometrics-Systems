"""
Keystroke Biometrics Verification API
Serves pre-trained models for user verification.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

from model_training import KeystrokeAuthenticator
# Import your custom modules
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor

app = Flask(__name__)
CORS(app)

# Configuration
MODELS_DIR = Path("models")

def format_frontend_data(frontend_data, user_id):
    """Convert frontend JSON to the format expected by the Feature Extractor"""
    return {
        "metadata": {"userId": user_id, "totalSessions": 1},
        "sessions": [{
            "sessionId": "live_verify",
            "userId": user_id,
            "data": [{**k, "phraseNumber": 1} for k in frontend_data]
        }]
    }

@app.route('/users', methods=['GET'])
def get_users():
    """Returns list of users who have trained models"""
    # Scans the models/ directory for files like 'model_user_alice.pkl'
    if not MODELS_DIR.exists():
        return jsonify([])
    
    users = []
    for f in MODELS_DIR.glob("model_*.pkl"):
        # Extract "user_alice" from "model_user_alice.pkl"
        uid = f.stem.replace("model_", "") 
        users.append(uid)
    
    return jsonify(sorted(users))

@app.route('/verify', methods=['POST'])
def verify_user():
    try:
        data = request.json
        user_id = data.get('userId')
        keystroke_data = data.get('keystrokeData')
        
        # 1. Load Model AND Scaler
        model_path = MODELS_DIR / f"model_{user_id}.pkl"
        if not model_path.exists():
            return jsonify({"status": "error", "message": "Model not found"}), 404
            
        # Initialize authenticator just to use its load method
        auth = KeystrokeAuthenticator(pd.DataFrame()) 
        scaler = auth.load_model(model_path) # <--- Get the saved scaler!
        model = auth.model
        reference_cols = auth.feature_columns
        
        # 2. Extract Features
        temp_file = f"temp_{user_id}.json"
        formatted = format_frontend_data(keystroke_data, user_id)
        with open(temp_file, 'w') as f: json.dump(formatted, f)
            
        try:
            extractor = KeystrokeFeatureExtractor(temp_file)
            extractor.load_data()
            live_features_df = extractor.extract_all_features()
            
            # 3. Preprocess
            preprocessor = KeystrokePreprocessor(live_features_df)
            
            # A. Align Columns (Add missing 0s)
            preprocessor.align_columns(reference_cols)
            
            # B. Apply the SAVED Scaler (Crucial Fix!)
            # Do NOT use fit_transform here. Use transform_with_scaler.
            if scaler:
                preprocessor.transform_with_scaler(scaler)
            else:
                # Fallback if you forgot to retrain (will likely fail)
                print("WARNING: No scaler found in model file. Retrain pipeline!")
                preprocessor.features_df.fillna(0, inplace=True)

            X = preprocessor.features_df.fillna(0).values
            
            # 4. Predict
            prediction = model.predict(X)[0]
            score = model.decision_function(X)[0]
            
            print(f"DEBUG: Score: {score:.4f} | Threshold: 0.0")

        finally:
            if os.path.exists(temp_file): os.remove(temp_file)

        return jsonify({
            "authenticated": bool(prediction == 1),
            "score": float(score)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Verification Server active. Serving models from: {MODELS_DIR.absolute()}")
    app.run(debug=True, port=5000)