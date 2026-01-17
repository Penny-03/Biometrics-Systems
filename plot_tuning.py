import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
from model_training import KeystrokeAuthenticator

# CONFIG
USER_FILE = "data/genuine/user_chiara.json"
IMPOSTOR_DIR = "data/impostors"
CONTAMINATION_RANGE = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def main():
    print(f"Generating Error vs. Strictness Plot for {USER_FILE}...")
    
    # 1. Load User Data
    extractor = KeystrokeFeatureExtractor(USER_FILE)
    extractor.load_data()
    feats = extractor.extract_all_features()
    
    pre = KeystrokePreprocessor(feats)
    pre.identify_feature_columns()
    pre.handle_outliers(method='iqr', threshold=3.5)
    X_train, scaler = pre.fit_transform_scaler()
    
    # 2. Load Impostor Data (for testing)
    imp_feats_list = []
    import glob
    from pathlib import Path
    for f in Path(IMPOSTOR_DIR).glob("*.json"):
        ie = KeystrokeFeatureExtractor(str(f))
        ie.load_data()
        imp_feats_list.append(ie.extract_all_features())
    
    imp_df = pd.concat(imp_feats_list, ignore_index=True)
    
    # Align Impostors
    pre_imp = KeystrokePreprocessor(imp_df)
    pre_imp.align_columns(pre.feature_columns)
    pre_imp.transform_with_scaler(scaler)
    X_imp = pre_imp.features_df
    
    # 3. The Loop
    eer_scores = []
    
    for c in CONTAMINATION_RANGE:
        auth = KeystrokeAuthenticator(X_train)
        # Using 600 trees as per your optimal config for Ferola
        auth.train_isolation_forest(contamination=c, n_estimators=400, random_state=42)
        
        # Evaluate
        metrics = auth.evaluate_with_impostors(X_imp, user_name="Tuning")
        
        # Calculate EER (average of FAR and FRR)
        eer = (metrics['FAR'] + metrics['FRR']) / 2
        eer_scores.append(eer)
        print(f"Contamination: {c} -> EER: {eer:.4f}")

    # 4. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(CONTAMINATION_RANGE, eer_scores, marker='o', linestyle='-', color='b')
    plt.title(f'Error vs. Strictness (User 2)')
    plt.xlabel('Strictness (Contamination Parameter)')
    plt.ylabel('Equal Error Rate (EER)')
    plt.grid(True)
    
    plt.savefig('img/error_vs_strictness_2.png')
    print("Plot saved to img/error_vs_strictness_2.png")

if __name__ == "__main__":
    main()