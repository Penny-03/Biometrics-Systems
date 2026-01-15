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
from sklearn.metrics import roc_curve, auc
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
    
    def authenticate(self, new_features, threshold=None):
        """Authenticate a new typing sample"""
        # Ensure input handles NaNs
        #to FIX needs a scaler
        if hasattr(new_features, 'fillna'):
            new_features = new_features.fillna(0)
        
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            prediction = self.model.predict([new_features])[0]
            return prediction == 1

    def evaluate_with_impostors(self, impostor_features_df, user_name="Unknown"):
        """
        Evaluate model with impostor data AND Plot ROC Curve
        """
        print("\n" + "="*60)
        print("EVALUATION WITH IMPOSTOR DATA")
        print("="*60)
        
        # 1. Prepare Data
        X_genuine = self.features_df[self.feature_columns].fillna(0).values
        y_genuine = np.ones(len(X_genuine)) # 1 = Genuine
        
        X_impostor_df = impostor_features_df.reindex(columns=self.feature_columns, fill_value=0)
        X_impostor_df = X_impostor_df.fillna(0)
        X_impostor = X_impostor_df.values
        y_impostor = np.zeros(len(X_impostor)) # 0 = Impostor
        
        # Combine
        X_test = np.vstack([X_genuine, X_impostor])
        y_test = np.concatenate([y_genuine, y_impostor])
        
        # 2. Get Predictions AND Scores
        # Predict: Returns 1 (Normal) or -1 (Anomaly) -> Convert to 1/0
        y_pred = self.model.predict(X_test)
        y_pred = (y_pred == 1).astype(int) 
        
        # Decision Function: Raw scores (Positive = Normal, Negative = Anomaly)
        y_scores = self.model.decision_function(X_test)
        
        # 3. Calculate Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / len(y_test)
        
        # 4. PLOT ROC CURVE
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Match Rate (FMR)')
            plt.ylabel('True Match Rate (1 - FNMR)')
            plt.title(f'ROC Curve - {user_name}')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save the plot
            filename = f"roc_curve_{user_name}.png"
            plt.savefig(filename)
            print(f"✓ ROC Curve saved as '{filename}'")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not plot ROC: {e}")

        # Return metrics
        return {
            'FAR': far,
            'FRR': frr,
            'Accuracy': accuracy,
            'AUC': roc_auc
        }
    
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
    
    def save_model(self, filename, scaler=None):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'scaler': scaler
        }
        
       
            
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

        # Return the scaler if it exists
        return model_data.get('scaler')
      
    
    def decision_score(self, new_features):
        if self.model_type == 'OneClassSVM':
            return self.model.decision_function([new_features])[0]
        elif self.model_type == 'IsolationForest':
            return self.model.decision_function([new_features])[0]
