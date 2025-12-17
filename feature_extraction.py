"""
Keystroke Dynamics - Feature Extraction and Preprocessing
Updated with enhanced features and proper train/test handling
"""

import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class KeystrokeFeatureExtractor:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.raw_data = None
        self.features_df = None
        
    def load_data(self):
        with open(self.json_file_path, 'r') as f:
            self.raw_data = json.load(f)
        print(f"✓ Loaded data for user: {self.raw_data['metadata']['userId']}")
        print(f"✓ Total sessions: {self.raw_data['metadata']['totalSessions']}")
        return self.raw_data

    def extract_phrase_features(self, phrase_keystrokes):
        """Extract enhanced features from a single phrase"""
        if len(phrase_keystrokes) < 2:
            return None

        dwell_times = [ks['dwellTime'] for ks in phrase_keystrokes]
        flight_times = [ks['flightTime'] for ks in phrase_keystrokes if ks['flightTime'] > 0]

        # UP-DOWN latencies (release -> next press)
        ud_latencies = []
        for i in range(len(phrase_keystrokes) - 1):
            ud = phrase_keystrokes[i + 1]['pressTime'] - phrase_keystrokes[i]['releaseTime']
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
        total_time = phrase_keystrokes[-1]['releaseTime'] - phrase_keystrokes[0]['pressTime']
        cps = (len(phrase_keystrokes) / total_time) * 1000 if total_time > 0 else 0

        features['typing_speed'] = cps
        features['total_keystrokes'] = len(phrase_keystrokes)
        features['speed_per_key_std'] = np.std(dwell_times) / (cps + 1e-6)

        # Ratios
        features['dwell_flight_ratio'] = features['dwell_mean'] / (features['flight_mean'] + 1)

        return features

    def extract_digraph_features(self, phrase_keystrokes, top_n=10):
        """Extract digraph timing features"""
        digraph_times = {}

        for i in range(len(phrase_keystrokes) - 1):
            key1 = phrase_keystrokes[i]['key'].replace(" ", "SPC").replace("[", "").replace("]", "")
            key2 = phrase_keystrokes[i + 1]['key'].replace(" ", "SPC").replace("[", "").replace("]", "")
            digraph = f"{key1}_{key2}"

            timing = phrase_keystrokes[i + 1]['pressTime'] - phrase_keystrokes[i]['pressTime']
            if timing >= 0:
                digraph_times.setdefault(digraph, []).append(timing)

        digraph_features = {}
        sorted_digraphs = sorted(digraph_times.items(), key=lambda x: len(x[1]), reverse=True)

        for digraph, times in sorted_digraphs[:top_n]:
            digraph_features[f"dig_{digraph}_mean"] = np.mean(times)
            digraph_features[f"dig_{digraph}_std"] = np.std(times)

        return digraph_features

    def extract_basic_features(self, session_data):
        """Extract features from all phrases in a session"""
        keystrokes = session_data['data']
        
        if len(keystrokes) == 0:
            return []
        
        # Group keystrokes by phraseNumber
        phrases = {}
        for ks in keystrokes:
            phrase_num = ks.get('phraseNumber', 1)
            if phrase_num not in phrases:
                phrases[phrase_num] = []
            phrases[phrase_num].append(ks)
        
        # Extract features for each phrase
        all_phrase_features = []
        for phrase_num in sorted(phrases.keys()):
            phrase_keystrokes = phrases[phrase_num]
            
            # Basic features
            phrase_features = self.extract_phrase_features(phrase_keystrokes)
            
            if phrase_features is not None:
                # Digraph features (optional - can be disabled for speed)
                digraph_features = self.extract_digraph_features(phrase_keystrokes, top_n=10)
                phrase_features.update(digraph_features)
                
                # Add metadata
                phrase_features['session_id'] = session_data['sessionId']
                phrase_features['phrase_number'] = phrase_num
                phrase_features['user_id'] = session_data['userId']
                phrase_features['phrases_in_session'] = session_data.get('phrasesCompleted', len(phrases))
                
                all_phrase_features.append(phrase_features)
        
        return all_phrase_features
    
    def extract_all_features(self):
        """Extract features from all sessions"""
        if self.raw_data is None:
            self.load_data()
        
        all_features = []
        for session in self.raw_data['sessions']:
            phrase_features_list = self.extract_basic_features(session)
            all_features.extend(phrase_features_list)
        
        self.features_df = pd.DataFrame(all_features)
        self.features_df.fillna(0, inplace=True)
        
        total_phrases = len(self.features_df)
        total_sessions = len(self.raw_data['sessions'])
        avg_phrases = total_phrases / total_sessions if total_sessions > 0 else 0
        
        print(f"\n✓ Extracted features from {total_phrases} phrases across {total_sessions} sessions")
        print(f"✓ Average phrases per session: {avg_phrases:.1f}")
        print(f"✓ Features per phrase: {len(self.features_df.columns) - 4}")
        
        return self.features_df
    
    def save_features(self, output_path):
        """Save extracted features to CSV"""
        if self.features_df is None:
            self.extract_all_features()
        self.features_df.to_csv(output_path, index=False)
        print(f"\n✓ Features saved to: {output_path}")
    
    def get_feature_summary(self):
        """Print summary statistics of extracted features"""
        if self.features_df is None:
            self.extract_all_features()
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"\nUser ID: {self.features_df['user_id'].iloc[0]}")
        print(f"Number of phrases: {len(self.features_df)}")
        print(f"Number of features: {len(feature_cols)}")
        
        return self.features_df[feature_cols].describe()


class KeystrokePreprocessor:
    """Preprocess features with proper train/test handling"""
    
    def __init__(self, features_df):
        self.features_df = features_df.copy()
        self.feature_columns = None
        self.scaler = None
        
    def identify_feature_columns(self):
        """Identify which columns are features (exclude metadata)"""
        self.feature_columns = [col for col in self.features_df.columns 
                               if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        return self.feature_columns
    
    def align_columns(self, reference_columns):
        """
        Force columns to match reference exactly (CRITICAL for test data)
        
        Args:
            reference_columns: List of column names from training data
        """
        # Add missing columns with 0
        for col in reference_columns:
            if col not in self.features_df.columns:
                self.features_df[col] = 0.0
        
        # Keep only reference columns and enforce order
        self.features_df = self.features_df[reference_columns]
        self.feature_columns = reference_columns
        
        return self.features_df
    
    def handle_outliers(self, method='iqr', threshold=3.0):
        """
        Remove outliers from features
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (higher = more lenient)
        """
        if self.feature_columns is None:
            self.identify_feature_columns()
        
        original_count = len(self.features_df)
        
        if method == 'iqr':
            # Only apply to basic features (not digraphs which are sparse)
            basic_cols = [c for c in self.feature_columns if 'dig_' not in c]
            mask = pd.Series(True, index=self.features_df.index)
            
            for col in basic_cols:
                Q1 = self.features_df[col].quantile(0.25)
                Q3 = self.features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_mask = (self.features_df[col] >= lower_bound) & (self.features_df[col] <= upper_bound)
                mask = mask & col_mask
            
            self.features_df = self.features_df[mask]
        
        removed = original_count - len(self.features_df)
        print(f"\n✓ Removed {removed} outlier phrases (Threshold: {threshold})")
        print(f"✓ Remaining phrases: {len(self.features_df)}")
        
        return self.features_df
    
    def fit_transform_scaler(self, method='standard'):
        """
        Fit scaler on training data and transform
        Use this ONLY for training data!
        
        Args:
            method: 'standard' (z-score) or 'minmax'
            
        Returns:
            Tuple of (transformed_df, fitted_scaler)
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if self.feature_columns is None:
            self.identify_feature_columns()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.features_df[self.feature_columns] = self.scaler.fit_transform(
            self.features_df[self.feature_columns]
        )
        
        print(f"\n✓ Scaler Fitted & Features Normalized ({method})")
        
        return self.features_df, self.scaler

    def transform_with_scaler(self, scaler):
        """
        Apply existing scaler to new data (DO NOT RE-FIT)
        Use this for test/authentication data!
        
        Args:
            scaler: Fitted scaler from training data
            
        Returns:
            Transformed dataframe
        """
        if self.feature_columns is None:
            self.identify_feature_columns()
        
        self.features_df[self.feature_columns] = scaler.transform(
            self.features_df[self.feature_columns]
        )
        
        print(f"\n✓ Features transformed with existing scaler")
        
        return self.features_df


# Example usage
if __name__ == "__main__":
    print("KEYSTROKE DYNAMICS - FEATURE EXTRACTION")
    print("=" * 60)
    
    # Extract features
    extractor = KeystrokeFeatureExtractor("keystroke_data.json")
    extractor.load_data()
    features_df = extractor.extract_all_features()
    
    # Show summary
    summary = extractor.get_feature_summary()
    print("\nFeature Statistics:")
    print(summary)
    
    # Save features
    extractor.save_features("keystroke_features.csv")
    
    # Preprocess
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    
    preprocessor = KeystrokePreprocessor(features_df)
    preprocessor.identify_feature_columns()
    
    # Remove outliers
    preprocessor.handle_outliers(method='iqr', threshold=3.0)
    
    # Fit scaler and transform
    processed_features, scaler = preprocessor.fit_transform_scaler(method='standard')
    
    print("\n✓ Feature extraction and preprocessing complete!")