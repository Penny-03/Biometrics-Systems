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
import os
warnings.filterwarnings('ignore')

class KeystrokeAuthenticator:
    
    def __init__(self, features_df):
        
        self.features_df = features_df
        self.model = None
        self.model_type = None
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['session_id', 'phrase_number', 'user_id', 'phrases_in_session']]
        
    def train_one_class_svm(self, nu=0.1, kernel='rbf', gamma='scale'):
        print("\n" + "="*60)
        print("TRAINING ONE-CLASS SVM")
        print("="*60)
        
        # Clean training data
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
    
    def train_isolation_forest(self, contamination=0.1, n_estimators=100, random_state=42):
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST")
        print("="*60)
        
        X_df = self.features_df[self.feature_columns].fillna(0)
        X = X_df.values
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
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
        if hasattr(new_features, 'fillna'):
            new_features = new_features.fillna(0)
        
        if self.model_type in ['OneClassSVM', 'IsolationForest']:
            prediction = self.model.predict([new_features])[0]
            return prediction == 1

    def evaluate_with_impostors(self, impostor_features_df, user_name="Unknown"):
        print("\n" + "="*60)
        print("EVALUATION WITH IMPOSTOR DATA")
        print("="*60)
        
        # Prepare Data
        X_genuine = self.features_df[self.feature_columns].fillna(0).values
        y_genuine = np.ones(len(X_genuine)) # 1 = Genuine
        
        X_impostor_df = impostor_features_df.reindex(columns=self.feature_columns, fill_value=0)
        X_impostor_df = X_impostor_df.fillna(0)
        X_impostor = X_impostor_df.values
        y_impostor = np.zeros(len(X_impostor)) # 0 = Impostor
        
        # Combine
        X_test = np.vstack([X_genuine, X_impostor])
        y_test = np.concatenate([y_genuine, y_impostor])
        
        # Get Predictions AND Scores
        y_pred = self.model.predict(X_test)
        y_pred = (y_pred == 1).astype(int) 
        
        # Decision Function
        y_scores = self.model.decision_function(X_test)
        
        n_genuine = len(X_genuine)
        gen_scores = y_scores[:n_genuine]
        imp_scores = y_scores[n_genuine:]
        
        # Calculate Mean and Standard Deviation
        mu_gen = np.mean(gen_scores)
        mu_imp = np.mean(imp_scores)
        sigma_gen = np.std(gen_scores)
        sigma_imp = np.std(imp_scores)
        
        # Calculate d'
        denominator = np.sqrt(sigma_gen**2 + sigma_imp**2)
        
        if denominator == 0:
            d_prime = 0.0
        else:
            d_prime = (np.sqrt(2) * abs(mu_gen - mu_imp)) / denominator
            
        print(f" d' (Separability Index): {d_prime:.4f}")
        
        # Calculate Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / len(y_test)
        
        # PLOT ROC CURVE
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
            
            # Save the plot into 'img' folder
            filename = os.path.join("img", f"roc_curve_{user_name}.png")
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
            'AUC': roc_auc,
            'd_prime': d_prime 
        }
    
    def plot_confusion_matrix(self, impostor_features_df, user_name="Unknown"):
        # Get predictions
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
        plt.title(f'Confusion Matrix - {self.model_type} ({user_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the plot into 'img' folder
        filename = os.path.join("img", f"confusion_matrix_{user_name}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved as '{filename}'")
        plt.close()
    
    def save_model(self, filename, scaler=None):
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
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']

        # Return the scaler
        return model_data.get('scaler')
      
    def decision_score(self, new_features):
        if self.model_type == 'OneClassSVM':
            return self.model.decision_function([new_features])[0]
        elif self.model_type == 'IsolationForest':
            return self.model.decision_function([new_features])[0]