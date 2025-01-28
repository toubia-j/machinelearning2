# classes.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
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


class Preprocessor(TransformerMixin, BaseEstimator):
    """Classe identique à celle utilisée pendant l'entraînement"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        
    def fit(self, X, y=None):
        X = pd.Series(X).astype(str)
        self.vectorizer.fit(X)
        return self
    
    def transform(self, X):
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