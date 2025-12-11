"""
Keystroke Authentication Demo
This script demonstrates how to authenticate users with the trained model
"""

from model_training import KeystrokeAuthenticator
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
import pandas as pd
import numpy as np
from pathlib import Path

def authenticate_user(json_file_path, model_path="keystroke_auth_model.pkl", use_multi_phrase=True, threshold=0.7):
    """
    Authenticate a user based on their typing sample
    
    Args:
        json_file_path: Path to JSON file with new typing data
        model_path: Path to trained model
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
    
    # Step 2: Load the trained model
    print("\n1. Loading trained model...")
    # Create a dummy dataframe just to initialize
    dummy_df = pd.DataFrame(columns=['user_id', 'session_id', 'phrase_number'])
    authenticator = KeystrokeAuthenticator(dummy_df)
    authenticator.load_model(model_path)
    print(f"   ✓ Model loaded: {authenticator.model_type}")
    
    # Step 3: Extract features from new typing sample
    print("\n2. Processing new typing sample...")
    extractor = KeystrokeFeatureExtractor(json_file_path)
    extractor.load_data()
    raw_features = extractor.extract_all_features()
    
    if len(raw_features) == 0:
        return {"error": "No valid phrases found", "authenticated": False}
    
    print(f"   ✓ Extracted features from {len(raw_features)} phrases")
    
    # Step 4: Preprocess features (same as training)
    print("\n3. Preprocessing features...")
    preprocessor = KeystrokePreprocessor(raw_features)
    preprocessor.identify_feature_columns()
    preprocessor.normalize_features(method='standard')
    processed_features = preprocessor.features_df
    
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
        
        # Authenticate each phrase and collect results
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


def batch_authenticate(json_files, model_path="keystroke_auth_model.pkl"):
    """
    Authenticate multiple users in batch
    
    Args:
        json_files: List of JSON file paths
        model_path: Path to trained model
        
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
        result = authenticate_user(json_file, model_path, use_multi_phrase=True)
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
    
    # Check if model exists
    if not Path("keystroke_auth_model.pkl").exists():
        print("\n❌ Error: Model file 'keystroke_auth_model.pkl' not found!")
        print("   Please train a model first by running 'pipeline.py'")
        sys.exit(1)
    
    # Interactive mode
    print("\nAuthentication Options:")
    print("1. Authenticate single user")
    print("2. Batch authenticate multiple users")
    print("3. Test with example")
    
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
                threshold = max(0.5, min(1.0, threshold))  # Clamp to valid range
            except:
                threshold = 0.7
        
        # Authenticate
        result = authenticate_user(
            json_file,
            model_path="keystroke_auth_model.pkl",
            use_multi_phrase=use_multi_phrase,
            threshold=threshold
        )
        
    elif choice == "2":
        # Batch authentication
        files_input = input("\nEnter JSON file paths (comma-separated): ").strip()
        json_files = [f.strip() for f in files_input.split(',')]
        
        results = batch_authenticate(json_files, model_path="keystroke_auth_model.pkl")
        
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
        # Example/test mode
        print("\n" + "="*70)
        print("TEST MODE")
        print("="*70)
        print("\nThis mode requires a test JSON file.")
        print("Example usage:")
        print("""
# Authenticate a user with their typing sample
result = authenticate_user(
    json_file_path="user_typing.json",
    model_path="keystroke_auth_model.pkl",
    use_multi_phrase=True,
    threshold=0.7
)

if result['authenticated']:
    print("User is genuine!")
else:
    print("User is an impostor!")
        """)
    
    else:
        print("\n❌ Invalid choice")
    
    print("\n" + "="*70)
    print("Authentication complete!")
    print("="*70)