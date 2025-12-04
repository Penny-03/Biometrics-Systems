"""
Keystroke Dynamics - Feature Extraction and Preprocessing
This script processes the collected keystroke data and extracts features for authentication
Updated to handle multiple phrases per session with phraseNumber
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KeystrokeFeatureExtractor:
    """Extract features from keystroke timing data"""
    
    def __init__(self, json_file_path):
        """
        Initialize with path to the collected JSON data
        
        Args:
            json_file_path: Path to the JSON file from the data collector
        """
        self.json_file_path = json_file_path
        self.raw_data = None
        self.features_df = None
        
    def load_data(self):
        """Load the JSON data file"""
        with open(self.json_file_path, 'r') as f:
            self.raw_data = json.load(f)
        print(f"✓ Loaded data for user: {self.raw_data['metadata']['userId']}")
        print(f"✓ Total sessions: {self.raw_data['metadata']['totalSessions']}")
        return self.raw_data
    
    def extract_phrase_features(self, phrase_keystrokes):
        """
        Extract features from a single phrase
        
        Args:
            phrase_keystrokes: List of keystroke data for one phrase
            
        Returns:
            Dictionary of extracted features
        """
        if len(phrase_keystrokes) == 0:
            return None
        
        # Extract timing arrays
        dwell_times = [ks['dwellTime'] for ks in phrase_keystrokes]
        flight_times = [ks['flightTime'] for ks in phrase_keystrokes if ks['flightTime'] > 0]
        
        features = {}
        
        # Dwell time statistics
        features['dwell_mean'] = np.mean(dwell_times)
        features['dwell_std'] = np.std(dwell_times)
        features['dwell_median'] = np.median(dwell_times)
        features['dwell_min'] = np.min(dwell_times)
        features['dwell_max'] = np.max(dwell_times)
        features['dwell_q25'] = np.percentile(dwell_times, 25)
        features['dwell_q75'] = np.percentile(dwell_times, 75)
        
        # Flight time statistics
        if len(flight_times) > 0:
            features['flight_mean'] = np.mean(flight_times)
            features['flight_std'] = np.std(flight_times)
            features['flight_median'] = np.median(flight_times)
            features['flight_min'] = np.min(flight_times)
            features['flight_max'] = np.max(flight_times)
            features['flight_q25'] = np.percentile(flight_times, 25)
            features['flight_q75'] = np.percentile(flight_times, 75)
        else:
            features['flight_mean'] = 0
            features['flight_std'] = 0
            features['flight_median'] = 0
            features['flight_min'] = 0
            features['flight_max'] = 0
            features['flight_q25'] = 0
            features['flight_q75'] = 0
        
        # Typing speed (characters per second)
        total_time = phrase_keystrokes[-1]['releaseTime'] - phrase_keystrokes[0]['pressTime']
        features['typing_speed'] = (len(phrase_keystrokes) / total_time) * 1000  # Convert to chars/sec
        
        # Total keystrokes in phrase
        features['total_keystrokes'] = len(phrase_keystrokes)
        
        # Ratio features
        features['dwell_flight_ratio'] = features['dwell_mean'] / (features['flight_mean'] + 1)
        
        return features
    
    def extract_basic_features(self, session_data):
        """
        Extract basic timing features from a single session
        Now handles multiple phrases per session
        
        Args:
            session_data: Dictionary containing keystroke data for one session
            
        Returns:
            List of feature dictionaries (one per phrase)
        """
        keystrokes = session_data['data']
        
        if len(keystrokes) == 0:
            return []
        
        # Group keystrokes by phraseNumber
        phrases = {}
        for ks in keystrokes:
            phrase_num = ks.get('phraseNumber', 1)  # Default to 1 if not present
            if phrase_num not in phrases:
                phrases[phrase_num] = []
            phrases[phrase_num].append(ks)
        
        # Extract features for each phrase
        all_phrase_features = []
        for phrase_num in sorted(phrases.keys()):
            phrase_keystrokes = phrases[phrase_num]
            phrase_features = self.extract_phrase_features(phrase_keystrokes)
            
            if phrase_features is not None:
                # Add metadata
                phrase_features['session_id'] = session_data['sessionId']
                phrase_features['phrase_number'] = phrase_num
                phrase_features['user_id'] = session_data['userId']
                phrase_features['phrases_in_session'] = session_data.get('phrasesCompleted', len(phrases))
                
                all_phrase_features.append(phrase_features)
        
        return all_phrase_features
    
    def extract_digraph_features(self, phrase_keystrokes, top_n=10):
        """
        Extract features from specific key pair (digraph) timings
        
        Args:
            phrase_keystrokes: List of keystroke data for one phrase
            top_n: Number of most common digraphs to extract
            
        Returns:
            Dictionary of digraph features
        """
        digraph_times = {}
        
        # Calculate digraph timings
        for i in range(len(phrase_keystrokes) - 1):
            key1 = phrase_keystrokes[i]['key']
            key2 = phrase_keystrokes[i + 1]['key']
            digraph = f"{key1}-{key2}"
            
            # Time between pressing key1 and pressing key2
            timing = phrase_keystrokes[i + 1]['pressTime'] - phrase_keystrokes[i]['pressTime']
            
            if digraph not in digraph_times:
                digraph_times[digraph] = []
            digraph_times[digraph].append(timing)
        
        # Get statistics for most common digraphs
        digraph_features = {}
        sorted_digraphs = sorted(digraph_times.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (digraph, times) in enumerate(sorted_digraphs[:top_n]):
            prefix = f"digraph_{i}"
            digraph_features[f"{prefix}_mean"] = np.mean(times)
            digraph_features[f"{prefix}_std"] = np.std(times)
            digraph_features[f"{prefix}_name"] = digraph
        
        return digraph_features
    
    def extract_all_features(self):
        """Extract features from all sessions and phrases"""
        if self.raw_data is None:
            self.load_data()
        
        all_features = []
        
        for session in self.raw_data['sessions']:
            # Extract features for each phrase in the session
            phrase_features_list = self.extract_basic_features(session)
            all_features.extend(phrase_features_list)
        
        self.features_df = pd.DataFrame(all_features)
        
        total_phrases = len(self.features_df)
        total_sessions = len(self.raw_data['sessions'])
        avg_phrases = total_phrases / total_sessions if total_sessions > 0 else 0
        
        print(f"\n✓ Extracted features from {total_phrases} phrases across {total_sessions} sessions")
        print(f"✓ Average phrases per session: {avg_phrases:.1f}")
        print(f"✓ Features per phrase: {len(self.features_df.columns) - 4}")  # Excluding metadata
        
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
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        
        # Exclude metadata columns
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        
        print(f"\nUser ID: {self.features_df['user_id'].iloc[0]}")
        print(f"Number of phrases: {len(self.features_df)}")
        print(f"Number of sessions: {self.features_df['session_id'].nunique()}")
        print(f"Number of features per phrase: {len(feature_cols)}")
        
        print("\nKey Feature Statistics:")
        print("-" * 60)
        for col in ['dwell_mean', 'flight_mean', 'typing_speed']:
            if col in self.features_df.columns:
                print(f"{col:20s}: {self.features_df[col].mean():.2f} ± {self.features_df[col].std():.2f}")
        
        return self.features_df[feature_cols].describe()


class KeystrokePreprocessor:
    """Preprocess features for model training"""
    
    def __init__(self, features_df):
        """
        Initialize with features dataframe
        
        Args:
            features_df: DataFrame with extracted features
        """
        self.features_df = features_df.copy()
        self.feature_columns = None
        self.scaler = None
        
    def identify_feature_columns(self):
        """Identify which columns are features (exclude metadata)"""
        self.feature_columns = [col for col in self.features_df.columns 
                               if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        return self.feature_columns
    
    def handle_outliers(self, method='iqr', threshold=1.5):
        """
        Remove outliers from features
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or Z-score threshold
        """
        if self.feature_columns is None:
            self.identify_feature_columns()
        
        original_count = len(self.features_df)
        
        if method == 'iqr':
            # IQR method
            for col in self.feature_columns:
                Q1 = self.features_df[col].quantile(0.25)
                Q3 = self.features_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.features_df = self.features_df[
                    (self.features_df[col] >= lower_bound) & 
                    (self.features_df[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.features_df[self.feature_columns]))
            self.features_df = self.features_df[(z_scores < threshold).all(axis=1)]
        
        removed = original_count - len(self.features_df)
        print(f"\n✓ Removed {removed} outlier phrases")
        print(f"✓ Remaining phrases: {len(self.features_df)}")
        
        return self.features_df
    
    def normalize_features(self, method='standard'):
        """
        Normalize feature values
        
        Args:
            method: 'standard' (z-score) or 'minmax'
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
        
        print(f"\n✓ Features normalized using {method} scaling")
        
        return self.features_df
    
    def split_data(self, test_size=0.3, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if self.feature_columns is None:
            self.identify_feature_columns()
        
        X = self.features_df[self.feature_columns]
        y = self.features_df['user_id']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split: {len(X_train)} training, {len(X_test)} testing phrases")
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Path to your collected JSON file
    json_file = "keystroke_data_user001_1234567890.json"
    
    print("KEYSTROKE DYNAMICS - FEATURE EXTRACTION")
    print("=" * 60)
    
    # Step 1: Extract features
    extractor = KeystrokeFeatureExtractor(json_file)
    extractor.load_data()
    features_df = extractor.extract_all_features()
    
    # Show feature summary
    summary = extractor.get_feature_summary()
    print("\nFeature Statistics:")
    print(summary)
    
    # Save features
    extractor.save_features("keystroke_features.csv")
    
    # Step 2: Preprocess features
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    
    preprocessor = KeystrokePreprocessor(features_df)
    preprocessor.identify_feature_columns()
    
    # Remove outliers
    preprocessor.handle_outliers(method='iqr', threshold=1.5)
    
    # Normalize features
    preprocessor.normalize_features(method='standard')
    
    # Split data (if you have multiple users)
    # X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.3)
    
    print("\n✓ Feature extraction and preprocessing complete!")
    print(f"✓ Processed features saved and ready for model training")