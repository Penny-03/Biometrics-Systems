"""
COMPLETE KEYSTROKE DYNAMICS AUTHENTICATION PIPELINE
Features: 
- Multi-user training with Isolation Forest
- Robust Data Normalization (Fixes the 50% accuracy bug)
- Feature Alignment
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
from model_training import KeystrokeAuthenticator

DATA_DIR = Path("data")
GENUINE_DIR = DATA_DIR / "genuine"
IMPOSTOR_DIR = DATA_DIR / "impostors"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    print_header("KEYSTROKE DYNAMICS AUTHENTICATION - FIXED PIPELINE")
    
    if not GENUINE_DIR.exists():
        print(f"❌ Error: Directory {GENUINE_DIR} not found.")
        return

    genuine_files = sorted(list(GENUINE_DIR.glob("*.json")))
    impostor_files = sorted(list(IMPOSTOR_DIR.glob("*.json")))
    
    if not genuine_files:
        print(f"❌ Error: No genuine user files found.")
        return

    print(f"✓ Found {len(genuine_files)} Genuine Users")
    print(f"✓ Found {len(impostor_files)} Dedicated Impostor Files")

    # PER-USER TUNING CONFIGURATION
    # Contamination: Higher = Stricter Model (Reject more "weird" self-typing to keep hackers out)
    # n_estimators: Higher = Smoother decision boundary (Good for messy data)
    USER_CONFIG = {
        'user_ferola': {'contamination': 0.21, 'n_estimators': 500},
        'user_chiara': {'contamination': 0.14, 'n_estimators': 400},
        'user_iga':    {'contamination': 0.08, 'n_estimators': 300},
    }
    
    # Default settings for any new users not in the list
    DEFAULT_CONFIG = {'contamination': 0.05, 'n_estimators': 300}
    
    overall_metrics = []

    for user_file in genuine_files:
        user_name = user_file.stem
        print_header(f"PROCESSING USER: {user_name}")
        
        # --- 1. EXTRACT (Genuine) ---
        print(f"-> Extracting features...")
        extractor = KeystrokeFeatureExtractor(str(user_file))
        extractor.load_data()
        features_df = extractor.extract_all_features()
        print(f"   ✓ Extracted {len(features_df)} phrases")

        # --- 2. PREPROCESS (Train) ---
        print(f"-> Preprocessing & Fitting Scaler...")
        preprocessor = KeystrokePreprocessor(features_df)
        preprocessor.identify_feature_columns()
        
        # Relaxed outliers
        preprocessor.handle_outliers(method='iqr', threshold=3.5)
        
        # FIT THE SCALER HERE
        processed_features, trained_scaler = preprocessor.fit_transform_scaler(method='standard')
        
        # Save the list of columns we used for training (The Master List)
        master_feature_list = preprocessor.feature_columns
        print(f"   ✓ Training with {len(master_feature_list)} features")
        
        # --- 3. TRAIN (Isolation Forest) ---
        print(f"-> Training Isolation Forest...")
        authenticator = KeystrokeAuthenticator(processed_features)
        
        # GET CONFIG FOR CURRENT USER
        config = USER_CONFIG.get(user_name, DEFAULT_CONFIG)
        
        print(f"   (Using parameters: {config})")
        authenticator.train_isolation_forest(
            contamination=config['contamination'], 
            n_estimators=config['n_estimators']
        )
            
        model_filename = MODELS_DIR / f"model_{user_name}.pkl"
        authenticator.save_model(str(model_filename), scaler=trained_scaler)
        
        # --- 4. EVALUATE (Attack) ---
        print(f"-> Evaluating against attacks...")
        
        other_genuine = [f for f in genuine_files if f != user_file]
        attack_files = impostor_files + other_genuine
        
        all_impostor_features = []
        
        for imp_file in attack_files:
            if not imp_file.exists(): continue
            
            # Extract Impostor Features
            imp_extractor = KeystrokeFeatureExtractor(str(imp_file))
            imp_extractor.load_data()
            imp_feats = imp_extractor.extract_all_features()
            
            # PREPROCESS (Test) - CRITICAL STEP
            imp_preprocessor = KeystrokePreprocessor(imp_feats)
            
            # A. ALIGN columns to match Genuine User exactly (Fill missing with 0)
            imp_preprocessor.align_columns(master_feature_list)
            
            # B. SCALE using the Genuine User's scaler (Do not re-fit!)
            imp_preprocessor.transform_with_scaler(trained_scaler)
            
            all_impostor_features.append(imp_preprocessor.features_df)
            
        if all_impostor_features:
            impostor_df = pd.concat(all_impostor_features, ignore_index=True)
            
            metrics = authenticator.evaluate_with_impostors(impostor_df)
            metrics['User'] = user_name
            overall_metrics.append(metrics)
            
            try:
                authenticator.plot_confusion_matrix(impostor_df)
                if os.path.exists('confusion_matrix.png'):
                    os.rename('confusion_matrix.png', f'confusion_matrix_{user_name}.png')
            except: pass

    # --- REPORT ---
    print_header("FINAL REPORT")
    if overall_metrics:
        results_df = pd.DataFrame(overall_metrics)
        results_df['EER'] = (results_df['FAR'] + results_df['FRR']) / 2
        
        print(results_df[['User', 'FAR', 'FRR', 'EER', 'Accuracy']].to_string(index=False))
        print(f"\nAVERAGE EER: {results_df['EER'].mean()*100:.2f}%")
        
        best = results_df.loc[results_df['EER'].idxmin()]
        print(f" Best User: {best['User']} (EER: {best['EER']*100:.2f}%)")

if __name__ == "__main__":
    main()