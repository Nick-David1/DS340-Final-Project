#!/usr/bin/env python3
"""
Advanced Model Techniques for Maximum Performance
--------------------------------------------------
Implements state-of-the-art techniques:
- Advanced neural networks (dropout, batch norm, residual connections)
- CatBoost and LightGBM
- Focal loss for imbalance
- Stacking ensemble
- Feature engineering
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import json
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, MODELS_DIR, RESULTS_DIR, LOGS_DIR, DEVICE
from data_preprocessing import DataModule

# Try importing CatBoost and LightGBM
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Installing CatBoost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost as cb
    CATBOOST_AVAILABLE = True

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("Installing LightGBM...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True

# Setup logging
log_file = os.path.join(LOGS_DIR, f"advanced_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== FOCAL LOSS ====================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - better than standard CrossEntropy"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==================== ADVANCED NEURAL NETWORK ====================

class AdvancedTabularNN(nn.Module):
    """
    State-of-the-art neural network for tabular data with:
    - Residual connections
    - Batch normalization
    - Dropout with varying rates
    - Label smoothing capability
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], 
                 dropout_rates=[0.5, 0.4, 0.3, 0.2], use_batch_norm=True,
                 use_residual=True, num_classes=2):
        super(AdvancedTabularNN, self).__init__()
        
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]) if use_batch_norm else nn.Identity())
            self.dropouts.append(nn.Dropout(dropout_rates[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input
        out = self.input_layer(x)
        out = self.input_bn(out)
        out = nn.ReLU()(out)
        
        # Hidden layers with optional residual connections
        for i, (hidden, bn, dropout) in enumerate(zip(self.hidden_layers, self.batch_norms, self.dropouts)):
            identity = out
            
            out = hidden(out)
            out = bn(out)
            out = nn.ReLU()(out)
            out = dropout(out)
            
            # Residual connection (if dimensions match)
            if self.use_residual and identity.shape == out.shape:
                out = out + identity
        
        # Output
        out = self.output_layer(out)
        return out

class AdvancedNNTrainer:
    """Trainer with advanced techniques"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or DEVICE
        self.model = self.model.to(self.device)
        self.best_val_f1 = 0
        self.best_model_state = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def train(self, X_train, y_train, X_val, y_val, 
             epochs=100, batch_size=64, learning_rate=1e-3,
             use_focal_loss=True, label_smoothing=0.1,
             early_stopping_patience=20, use_lr_scheduler=True):
        """Train with all advanced techniques"""
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            logger.info("Using Focal Loss for class imbalance")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info(f"Using CrossEntropy with label smoothing={label_smoothing}")
        
        # Setup optimizer with weight decay for regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        if use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6
            )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation
            val_loss, val_acc, val_f1, val_auc = self._validate(X_val, y_val, criterion)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            
            # Learning rate scheduling
            if use_lr_scheduler:
                scheduler.step(val_f1)
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with Val F1: {self.best_val_f1:.4f}")
    
    def _validate(self, X_val, y_val, criterion):
        """Validate the model"""
        self.model.eval()
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predicted_np = predicted.cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            val_acc = accuracy_score(y_val_np, predicted_np)
            _, _, val_f1, _ = precision_recall_fscore_support(y_val_np, predicted_np, average='weighted')
            val_auc = roc_auc_score(y_val_np, probs_np[:, 1])
        
        return val_loss, val_acc, val_f1, val_auc
    
    def evaluate(self, X_test, y_test):
        """Full evaluation"""
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_np = predicted.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # Metrics
        accuracy = accuracy_score(y_test, predicted_np)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted_np, average='weighted')
        auc = roc_auc_score(y_test, probs_np[:, 1])
        cm = confusion_matrix(y_test, predicted_np)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }

# ==================== MAIN EXPERIMENT RUNNER ====================

def run_advanced_experiments(variant_name='both_only'):
    """Run experiments with advanced techniques"""
    
    logger.info(f"\n{'='*80}")
    logger.info("ADVANCED MODEL TECHNIQUES - MAXIMUM PERFORMANCE")
    logger.info(f"Variant: {variant_name}")
    logger.info(f"{'='*80}")
    
    # Load data
    data_module = DataModule(variant_name=variant_name)
    data_module.setup()
    
    tabular_data = data_module.get_tabular_data()
    X_train, y_train = tabular_data['train']
    X_val, y_val = tabular_data['val']
    X_test, y_test = tabular_data['test']
    
    logger.info(f"\nDataset: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Class distribution - Accept: {np.mean(y_train)*100:.1f}%, Reject: {(1-np.mean(y_train))*100:.1f}%")
    
    results = {}
    
    # ==================== 1. CATBOOST ====================
    logger.info(f"\n{'='*60}")
    logger.info("1. Training CatBoost (Handles categorical features natively)")
    logger.info(f"{'='*60}")
    
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info(f"Balanced training set: {len(y_train_balanced)} samples")
    
    catboost_model = cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        early_stopping_rounds=30,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    
    catboost_model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    cb_pred = catboost_model.predict(X_test)
    cb_proba = catboost_model.predict_proba(X_test)
    cb_acc = accuracy_score(y_test, cb_pred)
    _, _, cb_f1, _ = precision_recall_fscore_support(y_test, cb_pred, average='weighted')
    cb_auc = roc_auc_score(y_test, cb_proba[:, 1])
    
    logger.info(f"CatBoost Results: Acc={cb_acc:.4f}, F1={cb_f1:.4f}, AUC={cb_auc:.4f}")
    
    results['catboost'] = {
        'accuracy': cb_acc,
        'f1_weighted': cb_f1,
        'auc': cb_auc,
        'confusion_matrix': confusion_matrix(y_test, cb_pred).tolist()
    }
    
    # Save model
    catboost_model.save_model(os.path.join(MODELS_DIR, f"catboost_{variant_name}.cbm"))
    
    # ==================== 2. LIGHTGBM ====================
    logger.info(f"\n{'='*60}")
    logger.info("2. Training LightGBM (Fast gradient boosting)")
    logger.info(f"{'='*60}")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)]
    )
    
    lgb_pred = lgb_model.predict(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    _, _, lgb_f1, _ = precision_recall_fscore_support(y_test, lgb_pred, average='weighted')
    lgb_auc = roc_auc_score(y_test, lgb_proba[:, 1])
    
    logger.info(f"LightGBM Results: Acc={lgb_acc:.4f}, F1={lgb_f1:.4f}, AUC={lgb_auc:.4f}")
    
    results['lightgbm'] = {
        'accuracy': lgb_acc,
        'f1_weighted': lgb_f1,
        'auc': lgb_auc,
        'confusion_matrix': confusion_matrix(y_test, lgb_pred).tolist()
    }
    
    # Save model
    with open(os.path.join(MODELS_DIR, f"lightgbm_{variant_name}.pkl"), 'wb') as f:
        pickle.dump(lgb_model, f)
    
    # ==================== 3. ADVANCED NEURAL NETWORK ====================
    logger.info(f"\n{'='*60}")
    logger.info("3. Training Advanced Neural Network")
    logger.info(f"{'='*60}")
    logger.info("   - Residual connections")
    logger.info("   - Batch normalization")
    logger.info("   - Dropout (50% → 40% → 30% → 20%)")
    logger.info("   - Focal loss for imbalance")
    logger.info("   - Learning rate scheduling")
    logger.info("   - Early stopping (patience=20)")
    
    nn_model = AdvancedTabularNN(
        input_dim=X_train.shape[1],
        hidden_dims=[512, 256, 128, 64],
        dropout_rates=[0.5, 0.4, 0.3, 0.2],
        use_batch_norm=True,
        use_residual=True
    )
    
    nn_trainer = AdvancedNNTrainer(nn_model)
    nn_trainer.train(
        X_train_balanced, y_train_balanced,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        use_focal_loss=True,
        early_stopping_patience=20
    )
    
    nn_results = nn_trainer.evaluate(X_test, y_test)
    logger.info(f"Advanced NN Results: Acc={nn_results['accuracy']:.4f}, "
               f"F1={nn_results['f1_weighted']:.4f}, AUC={nn_results['auc']:.4f}")
    
    results['advanced_nn'] = nn_results
    
    # Save model
    torch.save(nn_model.state_dict(), os.path.join(MODELS_DIR, f"advanced_nn_{variant_name}.pth"))
    
    # ==================== 4. STACKING ENSEMBLE ====================
    logger.info(f"\n{'='*60}")
    logger.info("4. Training Stacking Ensemble")
    logger.info(f"{'='*60}")
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Base models
    estimators = [
        ('catboost', catboost_model),
        ('lightgbm', lgb_model),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42))
    ]
    
    # Meta-learner
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    logger.info("Training stacking ensemble...")
    stacking.fit(X_train_balanced, y_train_balanced)
    
    stack_pred = stacking.predict(X_test)
    stack_proba = stacking.predict_proba(X_test)
    stack_acc = accuracy_score(y_test, stack_pred)
    _, _, stack_f1, _ = precision_recall_fscore_support(y_test, stack_pred, average='weighted')
    stack_auc = roc_auc_score(y_test, stack_proba[:, 1])
    
    logger.info(f"Stacking Results: Acc={stack_acc:.4f}, F1={stack_f1:.4f}, AUC={stack_auc:.4f}")
    
    results['stacking_ensemble'] = {
        'accuracy': stack_acc,
        'f1_weighted': stack_f1,
        'auc': stack_auc,
        'confusion_matrix': confusion_matrix(y_test, stack_pred).tolist()
    }
    
    # Save model
    with open(os.path.join(MODELS_DIR, f"stacking_{variant_name}.pkl"), 'wb') as f:
        pickle.dump(stacking, f)
    
    # ==================== SUMMARY ====================
    logger.info(f"\n{'='*80}")
    logger.info("ADVANCED MODELS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    logger.info("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_weighted'], reverse=True)
    
    for model_name, result in sorted_results:
        logger.info(f"{model_name:<30} {result['accuracy']:>10.4f} {result['f1_weighted']:>10.4f} {result['auc']:>10.4f}")
    
    # Best model
    best_name, best_result = sorted_results[0]
    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 BEST MODEL: {best_name}")
    logger.info(f"{'='*80}")
    logger.info(f"  Accuracy: {best_result['accuracy']:.4f}")
    logger.info(f"  F1 Score: {best_result['f1_weighted']:.4f}")
    logger.info(f"  AUC: {best_result['auc']:.4f}")
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f"advanced_models_{variant_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['both_only', 'either_or_both', 'all_data'])
    args = parser.parse_args()
    
    all_results = {}
    
    for variant in args.variants:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"VARIANT: {variant}")
        logger.info(f"{'#'*80}\n")
        
        results = run_advanced_experiments(variant)
        all_results[variant] = results
    
    # Final comparison
    logger.info(f"\n\n{'='*80}")
    logger.info("FINAL COMPARISON ACROSS ALL VARIANTS")
    logger.info(f"{'='*80}\n")
    
    for variant, results in all_results.items():
        best = max(results.items(), key=lambda x: x[1]['f1_weighted'])
        logger.info(f"{variant:<20} - Best: {best[0]:<20} Acc={best[1]['accuracy']:.4f} F1={best[1]['f1_weighted']:.4f} AUC={best[1]['auc']:.4f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ ALL ADVANCED EXPERIMENTS COMPLETE!")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()



