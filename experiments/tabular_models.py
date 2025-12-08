"""
Tabular Models for Furniture Classification
Implements various ML algorithms for tabular data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import pickle
import json

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

# Neural Network with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Model selection and evaluation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, log_loss
)
from sklearn.calibration import CalibratedClassifierCV

from config import CONFIG, DEVICE, MODELS_DIR, RANDOM_SEED
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class TabularNeuralNetwork(nn.Module):
    """Feed-forward neural network for tabular data"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3, num_classes: int = 2):
        super(TabularNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)

class NeuralNetworkClassifier:
    """Sklearn-compatible wrapper for PyTorch neural network"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 1e-3, batch_size: int = 32,
                 epochs: int = 100, dropout: float = 0.3,
                 device: str = None, early_stopping_patience: int = 15):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.device = device or str(DEVICE)
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.best_model_state = None
        
    def fit(self, X, y, X_val=None, y_val=None, class_weight=None):
        """Train the neural network"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = TabularNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout
        ).to(self.device)
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Handle class weights
        if class_weight is not None:
            weight_tensor = torch.FloatTensor(class_weight).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def _validate(self, X_val, y_val, criterion):
        """Validate the model"""
        self.model.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
        
        self.model.train()
        return val_loss
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()

class TabularModelTrainer:
    """Comprehensive trainer for tabular models"""
    
    def __init__(self, model_type: str, hyperparameters: Optional[Dict] = None):
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def create_model(self, input_dim: Optional[int] = None):
        """Create model instance based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                **self.hyperparameters
            )
        
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                random_state=RANDOM_SEED,
                **self.hyperparameters
            )
        
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed")
            return xgb.XGBClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='logloss',
                **self.hyperparameters
            )
        
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed")
            return lgb.LGBMClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=-1,
                **self.hyperparameters
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                random_state=RANDOM_SEED,
                **self.hyperparameters
            )
        
        elif self.model_type == 'extra_trees':
            return ExtraTreesClassifier(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                **self.hyperparameters
            )
        
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=1000,
                **self.hyperparameters
            )
        
        elif self.model_type == 'svm':
            return SVC(
                random_state=RANDOM_SEED,
                probability=True,
                **self.hyperparameters
            )
        
        elif self.model_type == 'naive_bayes':
            return GaussianNB(**self.hyperparameters)
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_jobs=-1,
                **self.hyperparameters
            )
        
        elif self.model_type == 'neural_network':
            if input_dim is None:
                raise ValueError("input_dim required for neural network")
            return NeuralNetworkClassifier(
                input_dim=input_dim,
                **self.hyperparameters
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def hyperparameter_search(self, X_train, y_train, param_grid: Dict,
                             cv: int = 5, scoring: str = 'f1',
                             search_type: str = 'grid') -> Dict:
        """Perform hyperparameter tuning"""
        logger.info(f"Starting hyperparameter search for {self.model_type}")
        
        # Create base model
        base_model = self.create_model(input_dim=X_train.shape[1] if self.model_type == 'neural_network' else None)
        
        # Choose search method
        if search_type == 'grid':
            searcher = GridSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1 if self.model_type != 'neural_network' else 1,
                verbose=1, return_train_score=True
            )
        else:
            searcher = RandomizedSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring,
                n_iter=20, n_jobs=-1 if self.model_type != 'neural_network' else 1,
                verbose=1, random_state=RANDOM_SEED, return_train_score=True
            )
        
        # Fit
        searcher.fit(X_train, y_train)
        
        # Store results
        self.best_params = searcher.best_params_
        self.cv_results = pd.DataFrame(searcher.cv_results_)
        self.model = searcher.best_estimator_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {searcher.best_score_:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': searcher.best_score_,
            'cv_results': self.cv_results
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
             class_weight=None) -> None:
        """Train model with best parameters"""
        if self.best_params:
            self.hyperparameters.update(self.best_params)
        
        # Handle class weights
        if class_weight is not None:
            if self.model_type in ['random_forest', 'decision_tree', 'extra_trees', 'logistic_regression']:
                self.hyperparameters['class_weight'] = 'balanced'
            elif self.model_type == 'xgboost':
                # Calculate scale_pos_weight for XGBoost
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                self.hyperparameters['scale_pos_weight'] = neg_count / pos_count
            elif self.model_type == 'lightgbm':
                self.hyperparameters['class_weight'] = 'balanced'
        
        # Create and train model
        self.model = self.create_model(input_dim=X_train.shape[1] if self.model_type == 'neural_network' else None)
        
        if self.model_type == 'neural_network':
            self.model.fit(X_train, y_train, X_val, y_val, class_weight=class_weight)
        else:
            self.model.fit(X_train, y_train)
        
        logger.info(f"Model {self.model_type} trained successfully")
    
    def evaluate(self, X_test, y_test) -> Dict:
        """Comprehensive evaluation"""
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Additional metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # AUC and log loss if probabilities available
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba[:, 1])
                logloss = log_loss(y_test, y_proba)
            except:
                auc = 0.0
                logloss = float('inf')
        else:
            auc = 0.0
            logloss = float('inf')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        elif self.model_type == 'logistic_regression' and hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_[0])
        
        results = {
            'accuracy': accuracy,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'mcc': mcc,
            'cohen_kappa': kappa,
            'auc': auc,
            'log_loss': logloss,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return results
    
    def save_model(self, path: str):
        """Save trained model"""
        model_data = {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'best_params': self.best_params,
            'model': self.model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load saved model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.hyperparameters = model_data['hyperparameters']
        self.best_params = model_data['best_params']
        self.model = model_data['model']
        
        logger.info(f"Model loaded from {path}")

def run_tabular_experiments(data_module, variant_name: str) -> Dict:
    """Run all tabular model experiments for a data variant"""
    results = {}
    
    # Get tabular data
    tabular_data = data_module.get_tabular_data()
    X_train, y_train = tabular_data['train']
    X_val, y_val = tabular_data['val']
    X_test, y_test = tabular_data['test']
    feature_names = tabular_data['feature_names']
    
    logger.info(f"\nTabular data shapes:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    logger.info(f"  Features: {len(feature_names)}")
    
    # Calculate class weights
    class_weights = data_module.get_class_weights().numpy()
    
    # Define models to test
    models_config = {
        'random_forest': {
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'xgboost': {
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        } if XGBOOST_AVAILABLE else None,
        'gradient_boosting': {
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        },
        'logistic_regression': {
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2']
            }
        },
        'decision_tree': {
            'param_grid': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'neural_network': {
            'param_grid': {
                'hidden_dims': [[128, 64], [256, 128, 64]],
                'learning_rate': [1e-3, 1e-4],
                'batch_size': [32, 64],
                'dropout': [0.2, 0.3]
            }
        }
    }
    
    # Remove None entries
    models_config = {k: v for k, v in models_config.items() if v is not None}
    
    # Train and evaluate each model
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} on {variant_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create trainer
            trainer = TabularModelTrainer(model_name)
            
            # Hyperparameter search
            if model_name == 'neural_network':
                # For neural network, we'll do manual hyperparameter selection
                best_params = {
                    'hidden_dims': [256, 128, 64],
                    'learning_rate': 1e-3,
                    'batch_size': 32,
                    'dropout': 0.3,
                    'epochs': 100
                }
                trainer.hyperparameters = best_params
                trainer.best_params = best_params
            else:
                # Grid search for other models
                search_results = trainer.hyperparameter_search(
                    X_train, y_train,
                    param_grid=config['param_grid'],
                    cv=5,
                    scoring='f1_weighted',
                    search_type='grid' if len(config['param_grid']) <= 100 else 'random'
                )
            
            # Train with best parameters
            trainer.train(X_train, y_train, X_val, y_val, class_weight=class_weights)
            
            # Evaluate
            test_results = trainer.evaluate(X_test, y_test)
            
            # Save model
            model_path = os.path.join(
                MODELS_DIR,
                f"{model_name}_{variant_name}.pkl"
            )
            trainer.save_model(model_path)
            
            # Store results
            results[model_name] = {
                'test_results': test_results,
                'best_params': trainer.best_params,
                'model_path': model_path,
                'feature_names': feature_names
            }
            
            # Log summary
            logger.info(f"\n{model_name} Test Results:")
            logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
            logger.info(f"  F1 (macro): {test_results['f1_macro']:.4f}")
            logger.info(f"  F1 (weighted): {test_results['f1_weighted']:.4f}")
            logger.info(f"  AUC: {test_results['auc']:.4f}")
            logger.info(f"  MCC: {test_results['mcc']:.4f}")
            logger.info(f"  Cohen's Kappa: {test_results['cohen_kappa']:.4f}")
            logger.info(f"  Confusion Matrix:\n{test_results['confusion_matrix']}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Test tabular models
    from data_preprocessing import DataModule
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing tabular models...")
    
    # Test on 'all_data' variant
    data_module = DataModule(variant_name='all_data')
    data_module.setup()
    
    # Run experiments
    results = run_tabular_experiments(data_module, 'all_data')
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TABULAR MODEL SUMMARY")
    logger.info("="*60)
    
    for model_name, result in results.items():
        if 'error' not in result:
            logger.info(f"\n{model_name}:")
            logger.info(f"  F1 (weighted): {result['test_results']['f1_weighted']:.4f}")
            logger.info(f"  AUC: {result['test_results']['auc']:.4f}")
    
    logger.info("\nTabular model testing complete!")
