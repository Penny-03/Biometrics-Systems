# Keystroke Dynamics Authentication System

A complete biometric authentication system based on typing patterns and keystroke timing analysis.

## Project Overview

This project implements a keystroke dynamics-based authentication system that identifies users based on their unique typing patterns. It includes:

- Web-based data collection interface
- Feature extraction from keystroke timing data
- Multiple machine learning models for authentication
- Complete evaluation framework

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Data

Open `keystroke-collector.html` in your web browser:
- Enter a unique User ID
- Type the displayed text 10-15 times
- Download the JSON file

Repeat this process for:
- **Genuine user**: The person you want to authenticate
- **Impostors** (optional): Other people for testing

### 3. Run the Complete Pipeline

```bash
python pipeline.py
```

Follow the prompts to:
1. Load your collected data
2. Extract features
3. Train the model
4. Evaluate performance (if you have impostor data)

## Project Structure

```
Biometrics-Systems/
├── keystroke-collector.html       # Data collection interface
├── feature_extraction.py          # Feature extraction & preprocessing
├── model_training.py              # Model training & authentication
├── pipeline.py                    # End-to-end pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

##  How It Works

### 1. Data Collection
- Captures keystroke timing events (press/release)
- Records dwell time (how long keys are held)
- Records flight time (time between keystrokes)

### 2. Feature Extraction
Extracts statistical features from timing data:
- **Dwell time statistics**: mean, std, median, min, max, quartiles
- **Flight time statistics**: same as above
- **Typing speed**: characters per second
- **Digraph patterns**: timing of specific key pairs

### 3. Preprocessing
- Removes outliers using IQR method
- Normalizes features using StandardScaler
- Prepares data for model training

### 4. Model Training
Three model options:

#### One-Class SVM (Recommended)
- Best for single-user authentication
- Learns patterns of genuine user
- Detects anomalies as impostors
```python
authenticator.train_one_class_svm(nu=0.1, kernel='rbf')
```

#### Isolation Forest
- Alternative anomaly detection
- Good for high-dimensional data
```python
authenticator.train_isolation_forest(contamination=0.1)
```

#### Distance-Based
- Simple template matching
- Fast but less accurate
```python
authenticator.train_distance_based(metric='euclidean')
```

### 5. Evaluation Metrics

- **FAR** (False Acceptance Rate): % of impostors accepted
- **FRR** (False Rejection Rate): % of genuine users rejected
- **EER** (Equal Error Rate): Point where FAR = FRR
- **Accuracy**: Overall correct predictions

**Target Performance:**
- EER < 5%
- FAR < 2%
- FRR < 10%

##  Usage Examples

### Train a Model

```python
from feature_extraction import KeystrokeFeatureExtractor, KeystrokePreprocessor
from model_training import KeystrokeAuthenticator

# Load and process data
extractor = KeystrokeFeatureExtractor('user_data.json')
features = extractor.extract_all_features()

preprocessor = KeystrokePreprocessor(features)
preprocessor.normalize_features()

# Train model
auth = KeystrokeAuthenticator(preprocessor.features_df)
auth.train_one_class_svm()
auth.save_model('model.pkl')
```

### Authenticate a New Sample

```python
from model_training import KeystrokeAuthenticator
from feature_extraction import KeystrokeFeatureExtractor

# Load model
auth = KeystrokeAuthenticator(None)
auth.load_model('model.pkl')

# Process new typing sample
new_sample = KeystrokeFeatureExtractor('new_typing.json')
new_features = new_sample.extract_all_features()

# Authenticate
feature_vector = new_features.iloc[0][auth.feature_columns]
is_genuine = auth.authenticate(feature_vector)

print(f"Authentication: {'GENUINE' if is_genuine else 'IMPOSTOR'}")
```

##  Data Requirements

### Minimum Requirements
- **Training**: 10-15 sessions per genuine user
- **Testing**: 5+ impostor sessions (optional but recommended)
- **Session length**: Complete the provided text (44 characters)



##  License

This project is for educational purposes. Use responsibly.

