"""
Keystroke Dynamics - Model Training and Authentication
This script trains authentication models and evaluates performance
"""

#tTO DO: ADD authenticate_multi_phrase

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
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['session_id', 'session_number', 'user_id']]
        
    def train_one_class_svm(self, nu=0.1, kernel='rbf', gamma='auto'):
        """
        Train One-Class SVM model (best for single-user authentication)
        
        Args:
            nu: Upper bound on fraction of outliers (0.1 = expect 10% outliers)
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient
            
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING ONE-CLASS SVM")
        print("="*60)
        
        X = self.features_df[self.feature_columns].values
        
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.model.fit(X)
        self.model_type = 'OneClassSVM'
        
        # Calculate training accuracy
        predictions = self.model.predict(X)
        accuracy = np.sum(predictions == 1) / len(predictions)
        
        print(f"\n✓ Model trained successfully")
        print(f"✓ Training samples: {len(X)}")
        print(f"✓ Features used: {len(self.feature_columns)}")
        print(f"✓ Training accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def train_isolation_forest(self, contamination=0.1, n_estimators=100):
        """
        Train Isolation Forest model
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
            
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST")
        print("="*60)
        
        X = self.features_df[self.feature_columns].values
        
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
        print(f"✓ Training samples: {len(X)}")
        print(f"✓ Features used: {len(self.feature_columns)}")
        print(f"✓ Training accuracy: {accuracy*100:.2f}%")
        
        return self.model
    
    def train_distance_based(self, metric='euclidean'):
        """
        Train simple distance-based authentication
        Creates a template from training data
        
        Args:
            metric: Distance metric ('euclidean', 'manhattan', 'mahalanobis')
        """
        print("\n" + "="*60)
        print("TRAINING DISTANCE-BASED MODEL")
        print("="*60)
        
        X = self.features_df[self.feature_columns].values
        
        # Create user template (mean of all training samples)
        self.template = np.mean(X, axis=0)
        self.template_std = np.std(X, axis=0)
        self.metric = metric
        self.model_type = 'DistanceBased'
        
        print(f"\n✓ Template created successfully")
        print(f"✓ Training samples: {len(X)}")
        print(f"✓ Features in template: {len(self.template)}")
        print(f"✓ Distance metric: {metric}")
        
        return self.template
    
    def authenticate(self, new_features, threshold=None):
        """
        Authenticate a new typing sample
        
        Args:
            new_features: Feature vector for new sample
            threshold: Decision threshold (auto-calculated if None)
            
        Returns:
            Boolean (True = Genuine, False = Impostor)
        """
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
    
    def evaluate_with_impostors(self, impostor_features_df):
        """
        Evaluate model with impostor data
        
        Args:
            impostor_features_df: DataFrame with features from impostor users
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATION WITH IMPOSTOR DATA")
        print("="*60)
        
        # Genuine samples (from training data)
        X_genuine = self.features_df[self.feature_columns].values
        y_genuine = np.ones(len(X_genuine))
        
        # Impostor samples
        impostor_feature_cols = [col for col in impostor_features_df.columns 
                                if col in self.feature_columns]
        X_impostor = impostor_features_df[impostor_feature_cols].values
        y_impostor = np.zeros(len(X_impostor))
        
        # Combine data
        X_test = np.vstack([X_genuine, X_impostor])
        y_test = np.concatenate([y_genuine, y_impostor])
        
        # Predict
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            y_pred = self.model.predict(X_test)
            y_pred = (y_pred == 1).astype(int)  # Convert -1/1 to 0/1
        
        # Calculate metrics
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
        
        print(f"\nGenuine samples: {len(X_genuine)}")
        print(f"Impostor samples: {len(X_impostor)}")
        print(f"\nRESULTS:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  FAR:       {far*100:.2f}%  (Impostors accepted)")
        print(f"  FRR:       {frr*100:.2f}%  (Genuine users rejected)")
        print(f"  EER:       ~{((far + frr)/2)*100:.2f}%")
        
        return metrics
    
    def plot_confusion_matrix(self, impostor_features_df):
        """Plot confusion matrix"""
        # Get predictions
        X_genuine = self.features_df[self.feature_columns].values
        impostor_feature_cols = [col for col in impostor_features_df.columns 
                                if col in self.feature_columns]
        X_impostor = impostor_features_df[impostor_feature_cols].values
        
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
        plt.show()
    
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
        
        print(f"\n✓ Model loaded from: {filename}")


# Example usage
if __name__ == "__main__":
    print("KEYSTROKE DYNAMICS - MODEL TRAINING")
    print("=" * 60)
    
    # Load preprocessed features
    features_df = pd.read_csv("keystroke_features.csv")
    
    # Initialize authenticator
    authenticator = KeystrokeAuthenticator(features_df)
    
    # Method 1: Train One-Class SVM (Recommended)
    authenticator.train_one_class_svm(nu=0.1, kernel='rbf')
    
    # Method 2: Train Isolation Forest (Alternative)
    # authenticator.train_isolation_forest(contamination=0.1)
    
    # Method 3: Train Distance-Based (Simple)
    # authenticator.train_distance_based(metric='euclidean')
    
    # Save the model
    authenticator.save_model("keystroke_model.pkl")
    
    # If you have impostor data, evaluate the model
    # impostor_features = pd.read_csv("impostor_features.csv")
    # metrics = authenticator.evaluate_with_impostors(impostor_features)
    # authenticator.plot_confusion_matrix(impostor_features)
    
    print("\n" + "="*60)
    print("✓ Model training complete!")
    print("✓ Model saved and ready for authentication")
    print("="*60)
    
    # Example: Authenticate a new sample
    # new_sample_features = [...]  # Feature vector from new typing session
    # is_genuine = authenticator.authenticate(new_sample_features)
    # print(f"Authentication result: {'Genuine' if is_genuine else 'Impostor'}")