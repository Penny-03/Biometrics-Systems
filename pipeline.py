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

    # ANONYMIZATION
    ANONYMOUS_MAP = {
        'user_iga':    'User 1',
        'user_chiara': 'User 2',
        'user_ferola': 'User 3'
    }

    # CONFIGURATION
    USER_CONFIG = {
        'user_iga':    {'contamination': 0.08, 'n_estimators': 300},
        'user_chiara': {'contamination': 0.14, 'n_estimators': 400},
        'user_ferola': {'contamination': 0.21, 'n_estimators': 600},
    }
    DEFAULT_CONFIG = {'contamination': 0.1, 'n_estimators': 300}
    
    overall_metrics = []

    for user_file in genuine_files:
        real_name = user_file.stem
        
        display_name = ANONYMOUS_MAP.get(real_name, real_name)
        
        print_header(f"PROCESSING: {display_name} ({real_name})")
        
        # EXTRACTING FEATURES
        extractor = KeystrokeFeatureExtractor(str(user_file))
        extractor.load_data()
        features_df = extractor.extract_all_features()

        # PREPROCESSING
        preprocessor = KeystrokePreprocessor(features_df)
        preprocessor.identify_feature_columns()
        preprocessor.handle_outliers(method='iqr', threshold=3.5)
        processed_features, trained_scaler = preprocessor.fit_transform_scaler(method='standard')
        master_feature_list = preprocessor.feature_columns
        
        # TRAINING
        authenticator = KeystrokeAuthenticator(processed_features)
        config = USER_CONFIG.get(real_name, DEFAULT_CONFIG)
        
        print(f"   (Using parameters: {config})")
        authenticator.train_isolation_forest(
            contamination=config['contamination'], 
            n_estimators=config['n_estimators']
        )
        
        # Save model
        model_filename = MODELS_DIR / f"model_{real_name}.pkl"
        authenticator.save_model(str(model_filename), scaler=trained_scaler)
        
        # EVALUATION WITH IMPOSTORS
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
            
            metrics = authenticator.evaluate_with_impostors(impostor_df, user_name=display_name)
            metrics['User'] = display_name
            overall_metrics.append(metrics)
            
            try:
                authenticator.plot_confusion_matrix(impostor_df, user_name=display_name)
            except Exception as e:
                print(f"Warning: Could not plot confusion matrix: {e}")

    # FINAL REPORT
    print_header("FINAL REPORT")
    if overall_metrics:
        results_df = pd.DataFrame(overall_metrics)
        results_df['EER'] = (results_df['FAR'] + results_df['FRR']) / 2
        
        print(results_df[['User', 'FAR', 'FRR', 'EER', 'Accuracy', 'd_prime']].to_string(index=False))
        
        print(f"\nAVERAGE EER: {results_df['EER'].mean()*100:.2f}%")
        
        print(f"AVERAGE d':   {results_df['d_prime'].mean():.4f}")
        
        best = results_df.loc[results_df['EER'].idxmin()]
        print(f" Best User: {best['User']} (EER: {best['EER']*100:.2f}%)")

if __name__ == "__main__":
    main()