"""
Keystroke Authentication Demo
Updated for enhanced features and proper scaler handling
"""

from model_training import KeystrokeAuthenticator
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def authenticate_user(json_file_path, model_path="keystroke_auth_model.pkl", 
                     scaler_path="scaler.pkl", use_multi_phrase=True, threshold=0.7):
    """
    Authenticate a user based on their typing sample
    
    Args:
        json_file_path: Path to JSON file with new typing data
        model_path: Path to trained model
        scaler_path: Path to saved scaler
        use_multi_phrase: Use multiple phrases for voting (more accurate)
        threshold: Proportion of phrases that must match (0.5-1.0)
        
    Returns:
        dict: Authentication result with details
    """
    
    print("\n" + "="*70)
    print("KEYSTROKE AUTHENTICATION")
    print("="*70)
    
    # Step 1: Check if files exist
    if not Path(json_file_path).exists():
        return {"error": "JSON file not found", "authenticated": False}
    
    if not Path(model_path).exists():
        return {"error": "Model file not found", "authenticated": False}
    
    if not Path(scaler_path).exists():
        return {"error": "Scaler file not found", "authenticated": False}
    
    # Step 2: Load the trained model and scaler
    print("\n1. Loading trained model and scaler...")
    dummy_df = pd.DataFrame(columns=['user_id', 'session_id', 'phrase_number'])
    authenticator = KeystrokeAuthenticator(dummy_df)
    authenticator.load_model(model_path)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
        fitted_scaler = scaler_data['scaler']
        training_columns = scaler_data['feature_columns']
    
    print(f"   ✓ Model loaded: {authenticator.model_type}")
    print(f"   ✓ Scaler loaded with {len(training_columns)} features")
    
    # Step 3: Extract features from new typing sample
    print("\n2. Processing new typing sample...")
    extractor = KeystrokeFeatureExtractor(json_file_path)
    extractor.load_data()
    raw_features = extractor.extract_all_features()
    
    if len(raw_features) == 0:
        return {"error": "No valid phrases found", "authenticated": False}
    
    print(f"   ✓ Extracted features from {len(raw_features)} phrases")
    
    # Step 4: Preprocess features (CRITICAL: same as training!)
    print("\n3. Preprocessing features...")
    preprocessor = KeystrokePreprocessor(raw_features)
    preprocessor.identify_feature_columns()
    
    # CRITICAL: Align columns to match training data
    preprocessor.align_columns(training_columns)
    
    # CRITICAL: Use fitted scaler (DO NOT REFIT!)
    preprocessor.transform_with_scaler(fitted_scaler)
    processed_features = preprocessor.features_df
    
    print(f"   ✓ Features aligned and normalized")
    
    # Step 5: Authenticate
    print("\n4. Authenticating...")
    
    if use_multi_phrase and len(processed_features) > 1:
        # Multi-phrase authentication (more robust)
        print(f"   Using multi-phrase authentication with threshold={threshold}")
        
        # Get feature vectors for all phrases
        features_list = [
            processed_features.iloc[i][authenticator.feature_columns].values
            for i in range(len(processed_features))
        ]
        
        # Authenticate each phrase
        phrase_results = []
        for i, features in enumerate(features_list):
            result = authenticator.authenticate(features)
            phrase_results.append(result)
            status = "✓ GENUINE" if result else "✗ IMPOSTOR"
            print(f"     Phrase {i+1}: {status}")
        
        # Final decision using majority voting
        is_genuine = authenticator.authenticate_multi_phrase(features_list, threshold)
        genuine_count = sum(phrase_results)
        genuine_ratio = genuine_count / len(phrase_results)
        
        result = {
            "authenticated": is_genuine,
            "method": "multi-phrase",
            "total_phrases": len(phrase_results),
            "genuine_phrases": genuine_count,
            "genuine_ratio": genuine_ratio,
            "threshold": threshold,
            "phrase_results": phrase_results
        }
        
    else:
        # Single-phrase authentication
        print("   Using single-phrase authentication")
        features = processed_features.iloc[0][authenticator.feature_columns].values
        is_genuine = authenticator.authenticate(features)
        
        result = {
            "authenticated": is_genuine,
            "method": "single-phrase",
            "total_phrases": 1
        }
    
    # Step 6: Display result
    print("\n" + "="*70)
    print("AUTHENTICATION RESULT")
    print("="*70)
    
    if result["authenticated"]:
        print("\n✅ AUTHENTICATED - User is GENUINE")
    else:
        print("\n❌ REJECTED - User appears to be an IMPOSTOR")
    
    print("\nDetails:")
    if result["method"] == "multi-phrase":
        print(f"  Method: Multi-phrase voting")
        print(f"  Phrases analyzed: {result['total_phrases']}")
        print(f"  Genuine matches: {result['genuine_phrases']}/{result['total_phrases']}")
        print(f"  Match ratio: {result['genuine_ratio']*100:.1f}%")
        print(f"  Required threshold: {result['threshold']*100:.1f}%")
    else:
        print(f"  Method: Single-phrase")
    
    print("="*70)
    
    return result


def batch_authenticate(json_files, model_path="keystroke_auth_model.pkl", 
                       scaler_path="scaler.pkl"):
    """
    Authenticate multiple users in batch
    
    Args:
        json_files: List of JSON file paths
        model_path: Path to trained model
        scaler_path: Path to saved scaler
        
    Returns:
        List of results
    """
    results = []
    
    print("\n" + "="*70)
    print("BATCH AUTHENTICATION")
    print("="*70)
    print(f"\nProcessing {len(json_files)} samples...")
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {json_file}")
        result = authenticate_user(json_file, model_path, scaler_path, use_multi_phrase=True)
        result['file'] = json_file
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("BATCH SUMMARY")
    print("="*70)
    
    authenticated = sum(1 for r in results if r.get('authenticated', False))
    rejected = len(results) - authenticated
    
    print(f"\nTotal samples: {len(results)}")
    print(f"Authenticated: {authenticated} ({authenticated/len(results)*100:.1f}%)")
    print(f"Rejected: {rejected} ({rejected/len(results)*100:.1f}%)")
    
    return results


# Example usage
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("KEYSTROKE DYNAMICS - AUTHENTICATION SYSTEM")
    print("="*70)
    
    # Check if required files exist
    if not Path("keystroke_auth_model.pkl").exists():
        print("\n❌ Error: Model file 'keystroke_auth_model.pkl' not found!")
        print("   Please train a model first by running 'python pipeline.py'")
        sys.exit(1)
    
    if not Path("scaler.pkl").exists():
        print("\n❌ Error: Scaler file 'scaler.pkl' not found!")
        print("   Please train a model first by running 'python pipeline.py'")
        sys.exit(1)
    
    # Interactive mode
    print("\nAuthentication Options:")
    print("1. Authenticate single user")
    print("2. Batch authenticate multiple users")
    print("3. Show usage example")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Single user authentication
        json_file = input("\nEnter path to user's typing JSON file: ").strip()
        
        # Authentication settings
        use_multi = input("Use multi-phrase authentication? (y/n) [default: y]: ").strip().lower() or 'y'
        use_multi_phrase = (use_multi == 'y')
        
        threshold = 0.7  # Default
        if use_multi_phrase:
            thresh_input = input("Enter voting threshold (0.5-1.0) [default: 0.7]: ").strip()
            try:
                threshold = float(thresh_input) if thresh_input else 0.7
                threshold = max(0.5, min(1.0, threshold))
            except:
                threshold = 0.7
        
        # Authenticate
        result = authenticate_user(
            json_file,
            model_path="keystroke_auth_model.pkl",
            scaler_path="scaler.pkl",
            use_multi_phrase=use_multi_phrase,
            threshold=threshold
        )
        
    elif choice == "2":
        # Batch authentication
        files_input = input("\nEnter JSON file paths (comma-separated): ").strip()
        json_files = [f.strip() for f in files_input.split(',')]
        
        results = batch_authenticate(
            json_files,
            model_path="keystroke_auth_model.pkl",
            scaler_path="scaler.pkl"
        )
        
        # Show detailed results
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)
        for i, result in enumerate(results, 1):
            status = "✅ GENUINE" if result.get('authenticated', False) else "❌ IMPOSTOR"
            print(f"\n{i}. {result.get('file', 'Unknown')}: {status}")
            if result.get('method') == 'multi-phrase':
                print(f"   Match ratio: {result.get('genuine_ratio', 0)*100:.1f}%")
    
    elif choice == "3":
        # Example usage
        print("\n" + "="*70)
        print("USAGE EXAMPLE")
        print("="*70)
        print("""
# Simple authentication
from authenticate import authenticate_user

result = authenticate_user(
    json_file_path="new_user.json",
    model_path="keystroke_auth_model.pkl",
    scaler_path="scaler.pkl",
    use_multi_phrase=True,
    threshold=0.7
)

if result['authenticated']:
    print("✅ User is genuine - Access granted")
    print(f"Confidence: {result['genuine_ratio']*100:.1f}%")
else:
    print("❌ User is impostor - Access denied")

# Batch authentication
from authenticate import batch_authenticate

results = batch_authenticate(
    json_files=["user1.json", "user2.json", "user3.json"],
    model_path="keystroke_auth_model.pkl",
    scaler_path="scaler.pkl"
)

for r in results:
    print(f"{r['file']}: {'GENUINE' if r['authenticated'] else 'IMPOSTOR'}")
        """)
    
    else:
        print("\n❌ Invalid choice")
    
    print("\n" + "="*70)
    print("Authentication complete!")
    print("="*70)