"""
COMPLETE KEYSTROKE DYNAMICS AUTHENTICATION PIPELINE
Run this script for end-to-end processing
Updated for phrase-based data collection
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Import our custom modules (make sure they're in the same directory)
# If running as separate files, uncomment these:
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
from model_training import KeystrokeAuthenticator

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    """Main pipeline execution"""
    
    print_header("KEYSTROKE DYNAMICS AUTHENTICATION - COMPLETE PIPELINE")
    
    # ========================================================================
    # STEP 1: DATA COLLECTION (Already done with HTML collector)
    # ========================================================================
    print_header("STEP 1: DATA COLLECTION")
    print("✓ Data collected using web-based collector")
    print("  Files needed:")
    print("  - JSON file from genuine user (training data)")
    print("  - JSON files from other users (impostor data) [optional]")
    print("\n  Note: Each file should contain multiple phrases per session")
    print("        (minimum 10 phrases recommended per session)")
    
    # Get file paths
    genuine_user_file = input("\nEnter path to genuine user JSON file: ").strip()
    
    if not Path(genuine_user_file).exists():
        print("❌ Error: File not found!")
        return
    
    # ========================================================================
    # STEP 2: FEATURE EXTRACTION
    # ========================================================================
    print_header("STEP 2: FEATURE EXTRACTION")
    
    # Extract features from genuine user data
    extractor = KeystrokeFeatureExtractor(genuine_user_file)
    extractor.load_data()
    features_df = extractor.extract_all_features()
    
    # Show summary
    print("\nFeature Summary:")
    print(extractor.get_feature_summary())
    
    # Save features
    features_output = "genuine_user_features.csv"
    extractor.save_features(features_output)
    
    # ========================================================================
    # STEP 3: PREPROCESSING
    # ========================================================================
    print_header("STEP 3: PREPROCESSING")
    
    preprocessor = KeystrokePreprocessor(features_df)
    preprocessor.identify_feature_columns()
    
    # Check if we have enough data
    if len(features_df) < 10:
        print(f"\n⚠️  Warning: Only {len(features_df)} phrases collected.")
        print("   For best results, collect at least 10 phrases (ideally 15-20)")
        proceed = input("   Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting. Please collect more data.")
            return
    
    # Remove outliers
    print("\nRemoving outliers...")
    preprocessor.handle_outliers(method='iqr', threshold=1.5)
    
    # Normalize features
    print("Normalizing features...")
    processed_features = preprocessor.normalize_features(method='standard')
    
    # Save preprocessed features
    processed_features.to_csv("processed_features.csv", index=False)
    print("✓ Preprocessed features saved to 'processed_features.csv'")
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    print_header("STEP 4: MODEL TRAINING")
    
    # Check if we have enough samples
    if len(processed_features) < 5:
        print("⚠️  Warning: Less than 5 training phrases after preprocessing.")
        print("   Model accuracy may be poor. Collect more data for better results!")
    
    authenticator = KeystrokeAuthenticator(processed_features)
    
    # Train model (One-Class SVM recommended)
    print("\nSelect model type:")
    print("1. One-Class SVM (Recommended - Best accuracy)")
    print("2. Isolation Forest (Good alternative)")
    print("3. Distance-Based (Simple and fast)")
    
    choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        authenticator.train_one_class_svm(nu=0.1, kernel='rbf')
    elif choice == "2":
        authenticator.train_isolation_forest(contamination=0.1)
    elif choice == "3":
        authenticator.train_distance_based(metric='euclidean')
    else:
        print("Invalid choice, using One-Class SVM")
        authenticator.train_one_class_svm(nu=0.1, kernel='rbf')
    
    # Save model
    model_filename = "keystroke_auth_model.pkl"
    authenticator.save_model(model_filename)
    
    # ========================================================================
    # STEP 5: EVALUATION (if impostor data available)
    # ========================================================================
    print_header("STEP 5: EVALUATION (OPTIONAL)")
    
    has_impostor = input("\nDo you have impostor data for testing? (y/n): ").strip().lower()
    
    if has_impostor == 'y':
        impostor_files = input("Enter impostor JSON file paths (comma-separated): ").strip().split(',')
        
        all_impostor_features = []
        
        for imp_file in impostor_files:
            imp_file = imp_file.strip()
            if Path(imp_file).exists():
                print(f"\nProcessing: {imp_file}")
                imp_extractor = KeystrokeFeatureExtractor(imp_file)
                imp_extractor.load_data()
                imp_features = imp_extractor.extract_all_features()
                
                # Apply same preprocessing
                imp_preprocessor = KeystrokePreprocessor(imp_features)
                imp_preprocessor.identify_feature_columns()
                imp_preprocessor.normalize_features(method='standard')
                
                all_impostor_features.append(imp_preprocessor.features_df)
            else:
                print(f"⚠️  File not found: {imp_file}")
        
        if all_impostor_features:
            impostor_df = pd.concat(all_impostor_features, ignore_index=True)
            
            print(f"\n✓ Loaded {len(impostor_df)} impostor phrases for evaluation")
            
            # Evaluate
            metrics = authenticator.evaluate_with_impostors(impostor_df)
            
            # Plot confusion matrix
            try:
                authenticator.plot_confusion_matrix(impostor_df)
            except Exception as e:
                print(f"Note: Could not generate plot: {e}")
            
            print("\n" + "="*70)
            print("EVALUATION RESULTS:")
            print("="*70)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric:20s}: {value*100:.2f}%")
                else:
                    print(f"{metric:20s}: {value}")
            
            # Interpret results
            print("\n" + "="*70)
            print("INTERPRETATION:")
            print("="*70)
            eer = (metrics['FAR'] + metrics['FRR']) / 2
            if eer < 0.05:
                print("✅ Excellent! EER < 5% - Production ready")
            elif eer < 0.10:
                print("✓ Good! EER < 10% - Acceptable for most applications")
            elif eer < 0.20:
                print("⚠️  Fair. EER < 20% - Consider collecting more data")
            else:
                print("❌ Poor. EER > 20% - Collect more data and retrain")
    
    # ========================================================================
    # STEP 6: DEPLOYMENT READY
    # ========================================================================
    print_header("STEP 6: DEPLOYMENT")
    
    print("\n✓ Pipeline complete! Your authentication system is ready.")
    print("\nGenerated files:")
    print(f"  1. {features_output} - Extracted features (one row per phrase)")
    print(f"  2. processed_features.csv - Preprocessed features")
    print(f"  3. {model_filename} - Trained model")
    
    print("\n" + "="*70)
    print("HOW TO USE THE MODEL:")
    print("="*70)
    print("""
from model_training import KeystrokeAuthenticator
from feature_extraction import KeystrokeFeatureExtractor

# Load model
auth = KeystrokeAuthenticator(None)
auth.load_model('keystroke_auth_model.pkl')

# Process new typing sample
new_sample = KeystrokeFeatureExtractor('new_sample.json')
new_sample.load_data()
new_features = new_sample.extract_all_features()

# Method 1: Authenticate single phrase
phrase_features = new_features.iloc[0][auth.feature_columns]
is_genuine = auth.authenticate(phrase_features)
print(f"Result: {'Genuine' if is_genuine else 'Impostor'}")

# Method 2: Authenticate with multiple phrases (MORE ROBUST)
# Use majority voting - more reliable
all_features = [new_features.iloc[i][auth.feature_columns].values 
                for i in range(len(new_features))]
is_genuine = auth.authenticate_multi_phrase(all_features, threshold=0.7)
print(f"Multi-phrase result: {'Genuine' if is_genuine else 'Impostor'}")
    """)
    
    print_header("NEXT STEPS")
    print("""
    1. ✅ Collect more data (15-20 phrases minimum, 30+ ideal)
    2. ✅ Test with different users as impostors
    3. ✅ Tune model parameters (nu, contamination, threshold)
    4. ✅ Use multi-phrase authentication for better accuracy
    5. ✅ Integrate into your application
    
    Target Performance:
    - EER (Equal Error Rate): < 5% ✓
    - FAR (False Accept Rate): < 2% ✓
    - FRR (False Reject Rate): < 10% ✓
    
    Tips for Best Results:
    - Use same keyboard/device for consistency
    - Collect data at similar times of day
    - Ensure natural, relaxed typing
    - Use multi-phrase authentication (3-5 phrases)
    - Retrain periodically with new data
    """)
    
    print("\n" + "="*70)
    print("✓ Setup complete! Your keystroke authentication system is ready.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()