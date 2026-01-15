"""
COMPLETE KEYSTROKE DYNAMICS AUTHENTICATION PIPELINE
Features: 
- Multi-user training with Isolation Forest
- Robust Data Normalization (Fixes the 50% accuracy bug)
- Feature Alignment
- Automatic ROC Curve and Confusion Matrix generation
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
IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def main():
    print_header("KEYSTROKE DYNAMICS AUTHENTICATION - ANONYMIZED REPORT")
    
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

    # --- ANONYMIZATION MAPPING ---
    # Map internal filenames to Report Labels
    # User 1 = Iga (The Sheep / Best)
    # User 2 = Chiara (The Lamb / Middle)
    # User 3 = Ferola (The Goat / High Variance)
    ANONYMOUS_MAP = {
        'user_iga':    'User 1',
        'user_chiara': 'User 2',
        'user_ferola': 'User 3'
    }

    # CONFIGURATION
    USER_CONFIG = {
        'user_ferola': {'contamination': 0.21, 'n_estimators': 500},
        'user_chiara': {'contamination': 0.14, 'n_estimators': 400},
        'user_iga':    {'contamination': 0.08, 'n_estimators': 300},
    }
    DEFAULT_CONFIG = {'contamination': 0.05, 'n_estimators': 300}
    
    overall_metrics = []

    for user_file in genuine_files:
        real_name = user_file.stem  # e.g., "user_iga"
        
        # Determine the display name for the Report/Plots
        display_name = ANONYMOUS_MAP.get(real_name, real_name)
        
        print_header(f"PROCESSING: {display_name} ({real_name})")
        
        # --- 1. EXTRACT ---
        extractor = KeystrokeFeatureExtractor(str(user_file))
        extractor.load_data()
        features_df = extractor.extract_all_features()

        # --- 2. PREPROCESS ---
        preprocessor = KeystrokePreprocessor(features_df)
        preprocessor.identify_feature_columns()
        preprocessor.handle_outliers(method='iqr', threshold=3.5)
        processed_features, trained_scaler = preprocessor.fit_transform_scaler(method='standard')
        master_feature_list = preprocessor.feature_columns
        
        # --- 3. TRAIN ---
        authenticator = KeystrokeAuthenticator(processed_features)
        config = USER_CONFIG.get(real_name, DEFAULT_CONFIG)
        
        print(f"   (Using parameters: {config})")
        authenticator.train_isolation_forest(
            contamination=config['contamination'], 
            n_estimators=config['n_estimators']
        )
        
        # Save model using REAL name (so the App still works!)
        model_filename = MODELS_DIR / f"model_{real_name}.pkl"
        authenticator.save_model(str(model_filename), scaler=trained_scaler)
        
        # --- 4. EVALUATE ---
        other_genuine = [f for f in genuine_files if f != user_file]
        attack_files = impostor_files + other_genuine
        
        all_impostor_features = []
        for imp_file in attack_files:
            if not imp_file.exists(): continue
            imp_extractor = KeystrokeFeatureExtractor(str(imp_file))
            imp_extractor.load_data()
            imp_feats = imp_extractor.extract_all_features()
            
            imp_preprocessor = KeystrokePreprocessor(imp_feats)
            imp_preprocessor.align_columns(master_feature_list)
            imp_preprocessor.transform_with_scaler(trained_scaler)
            all_impostor_features.append(imp_preprocessor.features_df)
            
        if all_impostor_features:
            impostor_df = pd.concat(all_impostor_features, ignore_index=True)
            
            # --- CRITICAL CHANGE HERE ---
            # Pass 'display_name' (User 1) so the plot Title and Filename are anonymized
            metrics = authenticator.evaluate_with_impostors(impostor_df, user_name=display_name)
            metrics['User'] = display_name # Save anonymous name to table
            overall_metrics.append(metrics)
            
            try:
                # Pass 'display_name' here too
                authenticator.plot_confusion_matrix(impostor_df, user_name=display_name)
            except Exception as e:
                print(f"Warning: Could not plot confusion matrix: {e}")

    # --- REPORT ---
    print_header("FINAL REPORT")
    if overall_metrics:
        results_df = pd.DataFrame(overall_metrics)
        results_df['EER'] = (results_df['FAR'] + results_df['FRR']) / 2
        
        print(results_df[['User', 'FAR', 'FRR', 'EER', 'Accuracy']].to_string(index=False))
        print(f"\nAVERAGE EER: {results_df['EER'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()