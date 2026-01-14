"""
Flask Server for Keystroke Biometrics Verification
FIXED: Proper feature extraction, scaling, and column alignment
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_FOLDER = 'models'
TARGET_PHRASE = "the quick brown fox jumps over the lazy dog"

# Ensure model folder exists
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    print(f"Created '{MODEL_FOLDER}' folder. Place your .pkl model files here.")

def extract_phrase_features(keystroke_data):
    """
    Extract EXACT same features as training
    CRITICAL: Must match feature_extraction.py exactly
    """
    if len(keystroke_data) < 2:
        return None
    
    # Extract timing arrays
    dwell_times = [ks['dwellTime'] for ks in keystroke_data]
    flight_times = [ks['flightTime'] for ks in keystroke_data if ks['flightTime'] > 0]  # FIXED: exclude zeros
    
    # UP-DOWN latencies (release -> next press)
    ud_latencies = []
    for i in range(len(keystroke_data) - 1):
        ud = keystroke_data[i + 1]['pressTime'] - keystroke_data[i]['releaseTime']
        if ud >= 0:
            ud_latencies.append(ud)
    
    features = {}
    
    # Dwell time statistics
    features['dwell_mean'] = np.mean(dwell_times)
    features['dwell_std'] = np.std(dwell_times)
    features['dwell_median'] = np.median(dwell_times)
    features['dwell_min'] = np.min(dwell_times)
    features['dwell_max'] = np.max(dwell_times)
    features['dwell_cv'] = features['dwell_std'] / (features['dwell_mean'] + 1e-6)
    
    # Flight time statistics
    if len(flight_times) > 0:
        features['flight_mean'] = np.mean(flight_times)
        features['flight_std'] = np.std(flight_times)
        features['flight_median'] = np.median(flight_times)
        features['flight_min'] = np.min(flight_times)
        features['flight_max'] = np.max(flight_times)
        features['flight_cv'] = features['flight_std'] / (features['flight_mean'] + 1e-6)
    else:
        features['flight_mean'] = 0
        features['flight_std'] = 0
        features['flight_median'] = 0
        features['flight_min'] = 0
        features['flight_max'] = 0
        features['flight_cv'] = 0
    
    # UP-DOWN latency statistics
    if len(ud_latencies) > 0:
        features['ud_mean'] = np.mean(ud_latencies)
        features['ud_std'] = np.std(ud_latencies)
        features['ud_median'] = np.median(ud_latencies)
    else:
        features['ud_mean'] = 0
        features['ud_std'] = 0
        features['ud_median'] = 0
    
    # Typing speed features
    total_time = keystroke_data[-1]['releaseTime'] - keystroke_data[0]['pressTime']
    cps = (len(keystroke_data) / total_time) * 1000 if total_time > 0 else 0
    
    features['typing_speed'] = cps
    features['total_keystrokes'] = len(keystroke_data)
    features['speed_per_key_std'] = np.std(dwell_times) / (cps + 1e-6)
    
    # Ratios
    features['dwell_flight_ratio'] = features['dwell_mean'] / (features['flight_mean'] + 1)
    
    return features

@app.route('/')
def index():
    """Serve the HTML file"""
    return send_from_directory('.', 'verification.html')

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get list of enrolled users"""
    users = []
    try:
        if not os.path.exists(MODEL_FOLDER):
            return jsonify({"users": [], "message": f"'{MODEL_FOLDER}' folder not found"})
        
        for filename in os.listdir(MODEL_FOLDER):
            if filename.endswith('.pkl'):
                user_id = filename[:-4]
                users.append({
                    "id": user_id,
                    "name": user_id.replace('_', ' ').title()
                })
        
        return jsonify({
            "users": users,
            "count": len(users),
            "message": f"Found {len(users)} enrolled user(s)"
        })
    except Exception as e:
        return jsonify({"error": str(e), "users": []}), 500

@app.route('/api/verify', methods=['POST'])
def verify_user():
    """Verify user identity based on keystroke pattern"""
    data = request.json
    claimed_user = data.get('claimedUser')
    keystrokes = data.get('keystrokes')
    
    if not claimed_user or not keystrokes:
        return jsonify({
            "authenticated": False, 
            "message": "Missing required data"
        }), 400
    
    model_path = os.path.join(MODEL_FOLDER, f"{claimed_user}.pkl")
    
    if not os.path.exists(model_path):
        return jsonify({
            "authenticated": False, 
            "message": f"No model found for user '{claimed_user}'"
        }), 404
    
    try:
        print(f"\n{'='*60}")
        print(f"Verifying user: {claimed_user}")
        print(f"Keystrokes received: {len(keystrokes)}")
        print(f"{'='*60}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_columns = model_data.get('feature_columns', [])
            model_type = model_data.get('model_type', 'Unknown')
            scaler = model_data.get('scaler', None)
            
            print(f"Model type: {model_type}")
            print(f"Expected features: {len(feature_columns)}")
            print(f"Scaler present: {scaler is not None}")
        else:
            model = model_data
            feature_columns = []
            model_type = 'Unknown'
            scaler = None
        
        if model is None:
            return jsonify({
                "authenticated": False,
                "message": "Invalid model file"
            }), 500
        
        # Extract features
        features_dict = extract_phrase_features(keystrokes)
        
        if features_dict is None:
            return jsonify({
                "authenticated": False,
                "message": "Insufficient keystroke data"
            }), 400
        
        print(f"\nExtracted features (sample):")
        for key in list(features_dict.keys())[:5]:
            print(f"  {key}: {features_dict[key]:.2f}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # CRITICAL: Align columns with training data
        if feature_columns:
            # Add missing columns with 0
            for col in feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
                    print(f"  Added missing column: {col}")
            
            # Keep only training columns in same order
            features_df = features_df[feature_columns]
        
        # Fill any NaN values
        features_df = features_df.fillna(0)
        
        # Convert to numpy
        feature_vector = features_df.values
        
        print(f"\nFeature vector shape: {feature_vector.shape}")
        print(f"Feature vector sample: {feature_vector[0][:5]}")
        
        # CRITICAL: Apply scaler if present
        if scaler is not None:
            print("\nApplying scaler...")
            feature_vector = scaler.transform(feature_vector)
            print(f"Scaled vector sample: {feature_vector[0][:5]}")
        
        # Predict
        prediction = model.predict(feature_vector)[0]
        
        # Get score
        try:
            score = model.decision_function(feature_vector)[0]
        except:
            score = 0
        
        # For Isolation Forest: 1 = genuine, -1 = impostor
        is_authenticated = (prediction == 1)
        
        # Calculate confidence
        if model_type == 'IsolationForest':
            # Isolation Forest: negative = outlier, positive = inlier
            # Typical range: -0.5 to 0.5
            confidence = min(max((score + 0.3) * 150, 0), 100)
        elif model_type == 'OneClassSVM':
            # One-Class SVM: similar but different scale
            confidence = min(max((score + 0.5) * 100, 0), 100)
        else:
            confidence = 50
        
        print(f"\nPrediction: {prediction}")
        print(f"Score: {score:.4f}")
        print(f"Authenticated: {is_authenticated}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            "authenticated": bool(is_authenticated),
            "confidence": round(confidence, 1),
            "user": claimed_user,
            "score": float(score),
            "modelType": model_type,
            "message": "Identity verified" if is_authenticated else "Authentication failed"
        })
    
    except Exception as e:
        import traceback
        print("\nERROR:")
        traceback.print_exc()
        return jsonify({
            "authenticated": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "modelFolder": MODEL_FOLDER,
        "modelsAvailable": len([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pkl')]) if os.path.exists(MODEL_FOLDER) else 0
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("KEYSTROKE BIOMETRICS VERIFICATION SERVER")
    print("="*60)
    print(f"\nModel folder: {MODEL_FOLDER}/")
    print(f"Place your trained .pkl model files in '{MODEL_FOLDER}/' folder")
    print(f"\nExample: {MODEL_FOLDER}/alice.pkl")
    print("\nIMPORTANT: Model must include 'scaler' for proper authentication!")
    print("\nServer starting on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)