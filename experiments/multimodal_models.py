"""
Multi-Modal Fusion Models for Furniture Classification
Combines image and tabular data using various fusion strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)

from config import CONFIG, DEVICE, NUM_WORKERS, MODELS_DIR
from image_models import CNNClassifier
from tabular_models import TabularNeuralNetwork

# Setup logging
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """Dataset for multi-modal learning"""
    
    def __init__(self, image_dataset, tabular_features, tabular_labels):
        self.image_dataset = image_dataset
        self.tabular_features = torch.FloatTensor(tabular_features)
        self.tabular_labels = torch.LongTensor(tabular_labels)
        
        assert len(self.image_dataset) == len(self.tabular_features)
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # Get image data
        image_data = self.image_dataset[idx]
        
        # Get tabular data
        tabular = self.tabular_features[idx]
        label = self.tabular_labels[idx]
        
        return {
            'image': image_data['image'],
            'tabular': tabular,
            'label': label
        }

class EarlyFusionModel(nn.Module):
    """Early fusion: Concatenate features before final classification"""
    
    def __init__(self, cnn_model_name: str = 'resnet50',
                 tabular_input_dim: int = 64,
                 fusion_hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super(EarlyFusionModel, self).__init__()
        
        # Image feature extractor (CNN backbone)
        self.cnn_backbone = CNNClassifier(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=True
        ).backbone
        
        # Get CNN feature dimension
        self.cnn_feature_dim = self._get_cnn_feature_dim(cnn_model_name)
        
        # Tabular feature projector
        self.tabular_projector = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Fusion classifier
        fusion_input_dim = self.cnn_feature_dim + 64  # CNN features + projected tabular
        layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fusion_classifier = nn.Sequential(*layers)
    
    def _get_cnn_feature_dim(self, model_name):
        """Get output dimension of CNN backbone"""
        if model_name == 'resnet50':
            return 2048
        elif model_name == 'efficientnet_b0':
            return 1280
        elif model_name == 'mobilenet_v3':
            return 1280
        elif model_name == 'densenet121':
            return 1024
        else:
            return 512
    
    def forward(self, image, tabular):
        # Extract image features
        image_features = self.cnn_backbone(image)
        
        # Project tabular features
        tabular_features = self.tabular_projector(tabular)
        
        # Concatenate features
        combined_features = torch.cat([image_features, tabular_features], dim=1)
        
        # Final classification
        output = self.fusion_classifier(combined_features)
        
        return output

class LateFusionModel(nn.Module):
    """Late fusion: Separate models, combine predictions"""
    
    def __init__(self, cnn_model_name: str = 'resnet50',
                 tabular_input_dim: int = 64,
                 tabular_hidden_dims: List[int] = [128, 64],
                 fusion_method: str = 'weighted_average',
                 num_classes: int = 2):
        super(LateFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Image model
        self.image_model = CNNClassifier(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=True
        )
        
        # Tabular model
        self.tabular_model = TabularNeuralNetwork(
            input_dim=tabular_input_dim,
            hidden_dims=tabular_hidden_dims,
            num_classes=num_classes
        )
        
        # Learnable fusion weights
        if fusion_method == 'weighted_average':
            self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)
        elif fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(num_classes * 2, num_classes),
                nn.ReLU(),
                nn.Linear(num_classes, 2),
                nn.Softmax(dim=1)
            )
    
    def forward(self, image, tabular):
        # Get predictions from both models
        image_logits = self.image_model(image)
        tabular_logits = self.tabular_model(tabular)
        
        if self.fusion_method == 'average':
            # Simple average
            output = (image_logits + tabular_logits) / 2
            
        elif self.fusion_method == 'weighted_average':
            # Weighted average with learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)
            output = weights[0] * image_logits + weights[1] * tabular_logits
            
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            combined = torch.cat([image_logits, tabular_logits], dim=1)
            attention_weights = self.attention(combined)
            output = (attention_weights[:, 0:1] * image_logits + 
                     attention_weights[:, 1:2] * tabular_logits)
            
        elif self.fusion_method == 'max':
            # Maximum fusion
            output = torch.max(image_logits, tabular_logits)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return output

class CrossModalAttentionFusion(nn.Module):
    """Advanced fusion using cross-modal attention"""
    
    def __init__(self, cnn_model_name: str = 'resnet50',
                 tabular_input_dim: int = 64,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 num_classes: int = 2):
        super(CrossModalAttentionFusion, self).__init__()
        
        # Feature extractors
        self.cnn_backbone = CNNClassifier(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=True
        ).backbone
        
        self.cnn_feature_dim = self._get_cnn_feature_dim(cnn_model_name)
        
        # Project features to common dimension
        self.image_projector = nn.Linear(self.cnn_feature_dim, hidden_dim)
        self.tabular_projector = nn.Linear(tabular_input_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def _get_cnn_feature_dim(self, model_name):
        if model_name == 'resnet50':
            return 2048
        elif model_name == 'efficientnet_b0':
            return 1280
        elif model_name == 'mobilenet_v3':
            return 1280
        elif model_name == 'densenet121':
            return 1024
        else:
            return 512
    
    def forward(self, image, tabular):
        # Extract and project features
        image_features = self.cnn_backbone(image)
        image_proj = self.image_projector(image_features).unsqueeze(1)
        tabular_proj = self.tabular_projector(tabular).unsqueeze(1)
        
        # Cross-modal attention (image attending to tabular)
        img_attended, _ = self.cross_attention(
            query=image_proj,
            key=tabular_proj,
            value=tabular_proj
        )
        
        # Cross-modal attention (tabular attending to image)
        tab_attended, _ = self.cross_attention(
            query=tabular_proj,
            key=image_proj,
            value=image_proj
        )
        
        # Combine attended features
        combined = torch.cat([img_attended, tab_attended], dim=2)
        
        # Self-attention for refinement
        refined, _ = self.self_attention(combined, combined, combined)
        
        # Global pooling and classification
        pooled = refined.squeeze(1)
        output = self.classifier(pooled)
        
        return output

class MultiModalTrainer:
    """Trainer for multi-modal models"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or DEVICE
        self.model = self.model.to(self.device)
        self.best_val_f1 = 0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module,
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, tabular)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        # Additional metrics
        mcc = matthews_corrcoef(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'mcc': mcc,
            'cohen_kappa': kappa,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 50, learning_rate: float = 1e-4,
             class_weights: Optional[torch.Tensor] = None,
             early_stopping_patience: int = 15) -> Dict:
        """Full training loop"""
        
        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,
                               weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, criterion, optimizer)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}")
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, criterion)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"AUC: {val_metrics['auc']:.4f}")
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Learning rate scheduling
            scheduler.step(val_metrics['f1'])
            
            # Early stopping
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                logger.info(f"New best model! F1: {self.best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with F1: {self.best_val_f1:.4f}")
        
        return self.training_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, tabular)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Additional metrics
        mcc = matthews_corrcoef(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        except:
            auc = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        
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
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        return results
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")

def run_multimodal_experiments(data_module, variant_name: str) -> Dict:
    """Run all multi-modal experiments"""
    results = {}
    
    # Get data
    multimodal_data = data_module.get_multimodal_data()
    tabular_data = multimodal_data['tabular']
    image_datasets = multimodal_data['image_datasets']
    
    # Create multi-modal datasets
    train_dataset = MultiModalDataset(
        image_datasets['train'],
        tabular_data['train'][0],
        tabular_data['train'][1]
    )
    
    val_dataset = MultiModalDataset(
        image_datasets['val'],
        tabular_data['val'][0],
        tabular_data['val'][1]
    )
    
    test_dataset = MultiModalDataset(
        image_datasets['test'],
        tabular_data['test'][0],
        tabular_data['test'][1]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.multimodal.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG.multimodal.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.multimodal.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Get class weights
    class_weights = data_module.get_class_weights()
    
    # Test different fusion strategies
    fusion_models = {
        'early_fusion': EarlyFusionModel(
            cnn_model_name='resnet50',
            tabular_input_dim=tabular_data['train'][0].shape[1],
            fusion_hidden_dims=[512, 256, 128],
            dropout=0.3
        ),
        'late_fusion_weighted': LateFusionModel(
            cnn_model_name='resnet50',
            tabular_input_dim=tabular_data['train'][0].shape[1],
            tabular_hidden_dims=[128, 64],
            fusion_method='weighted_average'
        ),
        'late_fusion_attention': LateFusionModel(
            cnn_model_name='resnet50',
            tabular_input_dim=tabular_data['train'][0].shape[1],
            tabular_hidden_dims=[128, 64],
            fusion_method='attention'
        ),
        'cross_attention': CrossModalAttentionFusion(
            cnn_model_name='resnet50',
            tabular_input_dim=tabular_data['train'][0].shape[1],
            hidden_dim=256,
            num_heads=8,
            dropout=0.3
        )
    }
    
    for model_name, model in fusion_models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} on {variant_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create trainer
            trainer = MultiModalTrainer(model)
            
            # Train
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=CONFIG.multimodal.epochs,
                learning_rate=CONFIG.multimodal.learning_rate,
                class_weights=class_weights,
                early_stopping_patience=CONFIG.multimodal.early_stopping_patience
            )
            
            # Evaluate
            test_results = trainer.evaluate(test_loader)
            
            # Save model
            model_path = os.path.join(
                MODELS_DIR,
                f"{model_name}_{variant_name}.pth"
            )
            trainer.save_model(model_path)
            
            # Store results
            results[model_name] = {
                'test_results': test_results,
                'training_history': history,
                'model_path': model_path
            }
            
            # Log summary
            logger.info(f"\n{model_name} Test Results:")
            logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
            logger.info(f"  F1 (macro): {test_results['f1_macro']:.4f}")
            logger.info(f"  F1 (weighted): {test_results['f1_weighted']:.4f}")
            logger.info(f"  AUC: {test_results['auc']:.4f}")
            logger.info(f"  MCC: {test_results['mcc']:.4f}")
            logger.info(f"  Confusion Matrix:\n{test_results['confusion_matrix']}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Test multi-modal models
    from data_preprocessing import DataModule
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing multi-modal models...")
    
    # Test on 'all_data' variant
    data_module = DataModule(variant_name='all_data')
    data_module.setup()
    
    # Run experiments (simplified for testing)
    results = run_multimodal_experiments(data_module, 'all_data')
    
    logger.info("\nMulti-modal model testing complete!")
