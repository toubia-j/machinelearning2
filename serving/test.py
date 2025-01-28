import os
import joblib
import numpy as np
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss, 
    mean_absolute_error, precision_score, recall_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from tabulate import tabulate

class DataLoader:
    @staticmethod
    def load_data(file_path: str) -> Tuple[pd.Series, np.ndarray, MultiLabelBinarizer]:
        """Load and clean data with multi-label support"""
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.dropna(subset=['description', 'categories'])
        df['description'] = df['description'].fillna('').str.strip()
        
        # Convert categories to binary matrix
        df['categories'] = df['categories'].astype(str)
        df['categories'] = df['categories'].str.split(', ')  # Assuming comma-separated labels
        
        # Create MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df['categories'])
        
        return df['description'], y, mlb

class Preprocessor(TransformerMixin, BaseEstimator):
    """Enhanced preprocessor with basic text cleaning"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        
    def fit(self, X, y=None):
        # Convert to pandas Series for consistent handling
        X = pd.Series(X).astype(str)
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X):
        # Ensure input is always treated as a pandas Series
        X = pd.Series(X).astype(str)
        return self.vectorizer.transform(X)

class ModelTrainer:
    def __init__(self, models: Dict[str, Any], metrics: Dict[str, Any]):
        self.models = models
        self.metrics = metrics
        self.results = []
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models with extended metrics"""
        with tqdm(total=len(self.models), desc="Training models") as model_pbar:
            for model_name, model in self.models.items():
                model_pbar.set_description(f"Training {model_name}")
                
                # Training
                model.fit(X_train, y_train)
                
                # Predictions
                predictions = model.predict(X_test)
                
                # Calculate probabilities for MAE (if available)
                prob_predictions = (model.predict_proba(X_test) 
                                    if hasattr(model, 'predict_proba') 
                                    else None)
                
                # Calculate all metrics
                metrics_results = {}
                for metric_name, metric_func in self.metrics.items():
                    try:
                        if metric_name == 'mae' and prob_predictions is not None:
                            metrics_results[metric_name] = mean_absolute_error(
                                y_test, prob_predictions[:, 1] if prob_predictions.shape[1] == 2 else prob_predictions
                            )
                        else:
                            metrics_results[metric_name] = metric_func(y_true=y_test, y_pred=predictions)
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {str(e)}")
                        metrics_results[metric_name] = np.nan
                
                self.results.append({
                    'model': model_name,
                    **metrics_results,
                    'instance': model
                })
                model_pbar.update(1)

class ExperimentRunner:
    def __init__(self, config: Dict):
        self.config = config
        
    def run(self, X, y):
        """Run complete experiment pipeline"""
        # Split data - keep original text data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=42
        )
        
        # Preprocess data - create transformed versions
        preprocessor = self.config['preprocessor']()
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Initialize and run trainer with transformed data
        trainer = ModelTrainer(self.config['models'], self.config['metrics'])
        trainer.train_and_evaluate(X_train_transformed, X_test_transformed, y_train, y_test)
        
        # Save models and preprocessor
        if 'save_dir' in self.config:
            save_models(
                trainer.results,
                preprocessor,
                self.config['mlb'],
                self.config['save_dir']
            )
        
        # Return original text data for evaluation
        return trainer.results, X_test, y_test

def predict_genre_from_path(model_path: str, synopsis: str, threshold: float = 0.5) -> dict:
    import os
    import joblib
    
    # Extract directory and filename components
    directory = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    
    # Extract timestamp from model filename
    try:
        filename_parts = filename.split('_')
        timestamp = '_'.join(filename_parts[-2:]).replace('.pkl', '')
    except:
        raise ValueError("Invalid model filename format. Expected format: 'modelname_YYYYMMDD_HHMMSS.pkl'")
    
    # Construct paths for preprocessor and label binarizer
    preprocessor_path = "saved_models/preprocessor_20250126_222927.pkl"
    mlb_path = "saved_models/mlb_20250126_222927.pkl"
    
    # Verify all required files exist
    for path in [model_path, preprocessor_path, mlb_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")
    
    # Load resources
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    mlb = joblib.load(mlb_path)
    
    # Preprocess and predict
    processed_text = preprocessor.transform([synopsis])
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_text)
        prediction = (probabilities >= threshold).astype(int)
    else:
        prediction = model.predict(processed_text)
        probabilities = None
    
    # Convert to human-readable labels
    predicted_labels = mlb.inverse_transform(prediction)
    
    return {
        'genres': list(predicted_labels[0]),
        'probabilities': dict(zip(mlb.classes_, probabilities[0])) if probabilities is not None else None
    }


def test_pred(description):
    print("t1")
    prediction = predict_genre_from_path(
        model_path= "saved_models/svm_20250126_222927.pkl",
        synopsis=description
    )
    print("affichage Prediction avec 1.6.1",prediction['genres'])
    return prediction

test_pred("A thrilling mystery about a detective solving crimes in Victorian London")