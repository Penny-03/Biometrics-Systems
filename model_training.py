"""
Keystroke Dynamics - Model Training and Authentication
This script trains authentication models and evaluates performance
Updated to handle phrase-based features and fill missing values (NaNs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class KeystrokeAuthenticator:
    """Train and evaluate keystroke authentication models"""
    
    def __init__(self, features_df):
        """
        Initialize with preprocessed features
        
        Args:
            features_df: DataFrame with extracted and preprocessed features
        """
        self.features_df = features_df
        self.model = None
        self.model_type = None
        # Exclude metadata columns to find the actual feature set
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        
    def train_one_class_svm(self, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Train One-Class SVM model (best for single-user authentication)
        """
        print("\n" + "="*60)
        print("TRAINING ONE-CLASS SVM")
        print("="*60)
        
        # Clean training data of any unexpected NaNs
        X_df = self.features_df[self.feature_columns].fillna(0)
        X = X_df.values
        
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.model.fit(X)
        self.model_type = 'OneClassSVM'
        
        # Calculate training accuracy
        predictions = self.model.predict(X)
        accuracy = np.sum(predictions == 1) / len(predictions)
        
        print(f"\n✓ Model trained successfully")
        print(f"✓ Training phrases: {len(X)}")
        print(f"✓ Features used: {len(self.feature_columns)}")
        print(f"✓ Training accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def train_isolation_forest(self, contamination=0.1, n_estimators=100):
        """Train Isolation Forest model"""
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST")
        print("="*60)
        
        X_df = self.features_df[self.feature_columns].fillna(0)
        X = X_df.values
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.model.fit(X)
        self.model_type = 'IsolationForest'
        
        # Calculate training accuracy
        predictions = self.model.predict(X)
        accuracy = np.sum(predictions == 1) / len(predictions)
        
        print(f"\n✓ Model trained successfully")
        print(f"✓ Training phrases: {len(X)}")
        print(f"✓ Features used: {len(self.feature_columns)}")
        print(f"✓ Training accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def train_distance_based(self, metric='euclidean'):
        """Train simple distance-based authentication"""
        print("\n" + "="*60)
        print("TRAINING DISTANCE-BASED MODEL")
        print("="*60)
        
        X_df = self.features_df[self.feature_columns].fillna(0)
        X = X_df.values
        
        # Create user template (mean of all training samples)
        self.template = np.mean(X, axis=0)
        self.template_std = np.std(X, axis=0)
        self.metric = metric
        self.model_type = 'DistanceBased'
        
        print(f"\n✓ Template created successfully")
        print(f"✓ Training phrases: {len(X)}")
        print(f"✓ Features in template: {len(self.template)}")
        print(f"✓ Distance metric: {metric}")
        
        return self.template
    
    def authenticate(self, new_features, threshold=None):
        """Authenticate a new typing sample"""
        # Ensure input handles NaNs
        if hasattr(new_features, 'fillna'):
            new_features = new_features.fillna(0)
        
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            prediction = self.model.predict([new_features])[0]
            return prediction == 1
        
        elif self.model_type == 'DistanceBased':
            # Calculate distance to template
            if self.metric == 'euclidean':
                distance = np.linalg.norm(new_features - self.template)
            elif self.metric == 'manhattan':
                distance = np.sum(np.abs(new_features - self.template))
            
            # Auto-calculate threshold if not provided
            if threshold is None:
                threshold = np.mean(self.template_std) * 3
            
            return distance < threshold
    
    def authenticate_multi_phrase(self, features_list, threshold=0.5):
        """Authenticate using multiple phrases (majority voting)"""
        if len(features_list) == 0:
            return False
        
        genuine_count = sum(1 for features in features_list if self.authenticate(features))
        genuine_ratio = genuine_count / len(features_list)
        
        return genuine_ratio >= threshold
    
    def evaluate_with_impostors(self, impostor_features_df):
        """
        Evaluate model with impostor data
        (CRITICAL FIX: Handles NaNs and mismatched columns)
        """
        print("\n" + "="*60)
        print("EVALUATION WITH IMPOSTOR DATA")
        print("="*60)
        
        # 1. Genuine samples (clean up training data)
        X_genuine = self.features_df[self.feature_columns].fillna(0).values
        y_genuine = np.ones(len(X_genuine))
        
        # 2. Impostor samples (CRITICAL FIX)
        # We enforce strict column alignment.
        # - If impostor is missing a column the model needs: fill with 0
        # - If impostor has extra columns: ignore them
        # - If there are NaNs: fill with 0
        X_impostor_df = impostor_features_df.reindex(columns=self.feature_columns, fill_value=0)
        X_impostor_df = X_impostor_df.fillna(0)
        X_impostor = X_impostor_df.values
        
        y_impostor = np.zeros(len(X_impostor))
        
        # 3. Combine data
        X_test = np.vstack([X_genuine, X_impostor])
        y_test = np.concatenate([y_genuine, y_impostor])
        
        # 4. Predict
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred == 1).astype(int)  # Convert -1/1 to 0/1
        
        # 5. Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'FAR': far,
            'FRR': frr,
            'Accuracy': accuracy,
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        }
        
        print(f"\nGenuine phrases: {len(X_genuine)}")
        print(f"Impostor phrases: {len(X_impostor)}")
        print(f"\nRESULTS:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  FAR:       {far*100:.2f}%  (Impostors accepted)")
        print(f"  FRR:       {frr*100:.2f}%  (Genuine users rejected)")
        print(f"  EER:       ~{((far + frr)/2)*100:.2f}%")
        
        return metrics
    
    def plot_confusion_matrix(self, impostor_features_df):
        """Plot confusion matrix"""
        # Get predictions (using same alignment logic as evaluate)
        X_genuine = self.features_df[self.feature_columns].fillna(0).values
        
        X_impostor_df = impostor_features_df.reindex(columns=self.feature_columns, fill_value=0)
        X_impostor_df = X_impostor_df.fillna(0)
        X_impostor = X_impostor_df.values
        
        X_test = np.vstack([X_genuine, X_impostor])
        y_test = np.concatenate([np.ones(len(X_genuine)), np.zeros(len(X_impostor))])
        
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred == 1).astype(int)
        
        # Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Impostor', 'Genuine'],
                   yticklabels=['Impostor', 'Genuine'])
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        # plt.show() 
    
    def save_model(self, filename):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }
        
        if self.model_type == 'DistanceBased':
            model_data['template'] = self.template
            model_data['template_std'] = self.template_std
            model_data['metric'] = self.metric
            
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to: {filename}")
    
    def load_model(self, filename):
        """Load trained model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        
        if self.model_type == 'DistanceBased':
            self.template = model_data['template']
            self.template_std = model_data['template_std']
            self.metric = model_data['metric']
            self.scaler = model_data.get('scaler', None)

        
        print(f"\n✓ Model loaded from: {filename}")
    
    def decision_score(self, new_features):
        if self.model_type == 'OneClassSVM':
            return self.model.decision_function([new_features])[0]
        elif self.model_type == 'IsolationForest':
            return self.model.decision_function([new_features])[0]
