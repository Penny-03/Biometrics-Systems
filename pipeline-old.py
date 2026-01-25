import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
from model_training import KeystrokeAuthenticator

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    
    print_header("KEYSTROKE DYNAMICS AUTHENTICATION - COMPLETE PIPELINE")

    print_header("STEP 1: DATA COLLECTION")
    print("✓ Data collected using web-based collector")
    print("  Files needed:")
    print("  - JSON file from genuine user (training data)")
    print("  - JSON files from other users (impostor data) [optional]")
    
    genuine_user_file = input("\nEnter path to genuine user JSON file: ").strip()
    
    if not Path(genuine_user_file).exists():
        print("❌ Error: File not found!")
        return

    print_header("STEP 2: FEATURE EXTRACTION")
    
    extractor = KeystrokeFeatureExtractor(genuine_user_file)
    extractor.load_data()
    features_df = extractor.extract_all_features()
    
    print("\nFeature Summary:")
    print(extractor.get_feature_summary())
    
    features_output = "genuine_user_features.csv"
    extractor.save_features(features_output)
    
    print_header("STEP 3: PREPROCESSING")
    
    preprocessor = KeystrokePreprocessor(features_df)
    preprocessor.identify_feature_columns()
    
    if len(features_df) < 10:
        print(f"\n⚠️  Warning: Only {len(features_df)} phrases collected.")
        print("   For best results, collect at least 10 phrases (ideally 15-20)")
        proceed = input("   Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Exiting. Please collect more data.")
            return
    
    # Remove outliers
    print("\nRemoving outliers...")
    preprocessor.handle_outliers(method='iqr', threshold=3.0)
    
    # Fit scaler and transform
    print("Normalizing features...")
    processed_features, fitted_scaler = preprocessor.fit_transform_scaler(method='standard')
    
    # Save preprocessed features
    processed_features.to_csv("processed_features.csv", index=False)
    print("✓ Preprocessed features saved to 'processed_features.csv'")
    
    # Save the scaler and feature columns
    scaler_data = {
        'scaler': fitted_scaler,
        'feature_columns': preprocessor.feature_columns
    }
    with open("scaler.pkl", 'wb') as f:
        pickle.dump(scaler_data, f)
    print("✓ Scaler saved to 'scaler.pkl'")

    print_header("STEP 4: MODEL TRAINING")
    
    if len(processed_features) < 5:
        print("⚠️  Warning: Less than 5 training phrases after preprocessing.")
        print("   Model accuracy may be poor. Collect more data for better results!")
    
    authenticator = KeystrokeAuthenticator(processed_features)
    
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
    
    model_filename = "keystroke_auth_model.pkl"
    authenticator.save_model(model_filename)
    
    print_header("STEP 5: EVALUATION (OPTIONAL)")
    
    has_impostor = input("\nDo you have impostor data for testing? (y/n): ").strip().lower()
    
    if has_impostor == 'y':
        impostor_files = input("Enter impostor JSON file paths (comma-separated): ").strip().split(',')
        
        all_impostor_features = []
        
        for imp_file in impostor_files:
            imp_file = imp_file.strip()
            if Path(imp_file).exists():
                print(f"\nProcessing: {imp_file}")
                
                # Extract features
                imp_extractor = KeystrokeFeatureExtractor(imp_file)
                imp_extractor.load_data()
                imp_features = imp_extractor.extract_all_features()
                
                # Preprocess with same scaler
                imp_preprocessor = KeystrokePreprocessor(imp_features)
                imp_preprocessor.identify_feature_columns()
                
                # Align columns first then transform
                imp_preprocessor.align_columns(preprocessor.feature_columns)
                imp_preprocessor.transform_with_scaler(fitted_scaler)
                
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
    
    print_header("STEP 6: DEPLOYMENT")
    
    print("\n✓ Pipeline complete! Your authentication system is ready.")
    print("\nGenerated files:")
    print(f"  1. {features_output} - Extracted features")
    print(f"  2. processed_features.csv - Preprocessed features")
    print(f"  3. {model_filename} - Trained model")
    print(f"  4. scaler.pkl - Feature scaler (CRITICAL for authentication!)")
    
    print("\n" + "="*70)
    print("HOW TO USE THE MODEL:")
    print("="*70)
    print("""
Use the authenticate.py script for easy authentication:
    python authenticate.py

Or programmatically:
    from authenticate import authenticate_user
    
    result = authenticate_user(
        json_file_path="new_user.json",
        model_path="keystroke_auth_model.pkl",
        scaler_path="scaler.pkl"
    )
    
    if result['authenticated']:
        print("✅ Access Granted")
    else:
        print("❌ Access Denied")
    """)
    
    print_header("NEXT STEPS")
    print("""
    1. ✅ Collect more data (15-20 phrases minimum, 30+ ideal)
    2. ✅ Test with different users as impostors
    3. ✅ Use multi-phrase authentication (3-5 phrases)
    4. ✅ Monitor and retrain periodically
    
    Enhanced Features Now Include:
    - UP-DOWN latencies (release-to-press timing)
    - Coefficient of variation (typing stability)
    - Speed stability metrics
    - Enhanced digraph features
    
    Target Performance:
    - EER (Equal Error Rate): < 5% ✓
    - FAR (False Accept Rate): < 2% ✓
    - FRR (False Reject Rate): < 10% ✓
    """)
    
    print("\n" + "="*70)
    print("✓ Setup complete! Your authentication system is ready.")
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