"""
CNN Models for Image-Only Furniture Classification
Implements various CNN architectures with transfer learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import logging
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from config import CONFIG, DEVICE, NUM_WORKERS, MODELS_DIR
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class CNNClassifier(nn.Module):
    """Base CNN classifier with transfer learning support"""
    
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 2, 
                 pretrained: bool = True, freeze_backbone: bool = False):
        super(CNNClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            self.feature_dim = num_features
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = num_features
            
        elif model_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = num_features
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = num_features
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Custom classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Option to freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features without classification"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

class ImageModelTrainer:
    """Trainer for CNN models"""
    
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
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        epoch_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # Calculate AUC if we have both classes
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 50, learning_rate: float = 1e-4,
             class_weights: Optional[torch.Tensor] = None,
             early_stopping_patience: int = 10,
             freeze_epochs: int = 0) -> Dict:
        """Full training loop with early stopping"""
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=CONFIG.image_model.weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Unfreeze backbone after freeze_epochs
            if epoch == freeze_epochs and freeze_epochs > 0:
                logger.info("Unfreezing backbone layers")
                self.model.unfreeze_backbone()
                # Adjust learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            
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
            
            # Early stopping based on F1 score
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
        """Comprehensive evaluation on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # AUC
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        except:
            auc = 0.0
        
        # Confusion matrix
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
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        return results
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model.model_name,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', 0)
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {path}")

def run_image_experiments(data_module, variant_name: str) -> Dict:
    """Run all CNN experiments for a data variant"""
    results = {}
    
    # Get image datasets
    from data_preprocessing import DataModule
    image_transforms = data_module._get_image_transforms()
    image_datasets = data_module.get_image_datasets(
        train_transform=image_transforms['train'],
        val_transform=image_transforms['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        image_datasets['train'],
        batch_size=CONFIG.image_model.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        image_datasets['val'],
        batch_size=CONFIG.image_model.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        image_datasets['test'],
        batch_size=CONFIG.image_model.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Get class weights
    class_weights = data_module.get_class_weights()
    
    # Test different architectures
    architectures = ['resnet50', 'efficientnet_b0', 'mobilenet_v3', 'densenet121']
    
    for model_name in architectures:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} on {variant_name}")
        logger.info(f"{'='*60}")
        
        # Create model
        model = CNNClassifier(
            model_name=model_name,
            num_classes=2,
            pretrained=True,
            freeze_backbone=True  # Start with frozen backbone
        )
        
        # Create trainer
        trainer = ImageModelTrainer(model)
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=CONFIG.image_model.epochs,
            learning_rate=CONFIG.image_model.learning_rate,
            class_weights=class_weights,
            early_stopping_patience=CONFIG.image_model.early_stopping_patience,
            freeze_epochs=CONFIG.image_model.freeze_backbone_epochs
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
        logger.info(f"  Confusion Matrix:\n{test_results['confusion_matrix']}")
    
    return results

if __name__ == "__main__":
    # Test the image models
    from data_preprocessing import DataModule
    
    logger.info("Testing CNN models...")
    
    # Test on small subset with 'all_data' variant
    data_module = DataModule(variant_name='all_data')
    data_module.setup()
    
    # Run experiments
    results = run_image_experiments(data_module, 'all_data')
    
    logger.info("\nImage model testing complete!")
