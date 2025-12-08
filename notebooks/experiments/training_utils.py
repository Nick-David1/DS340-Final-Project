"""
**AI-Generated based on class examples** Training utilities for furniture classification experiments
Common training loops, evaluation functions, and metrics
"""

import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    # Add progress bar
    pbar = tqdm(loader, desc=desc, leave=False)
    
    # Training loop
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute average metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, f1


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        # Eval loop
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute average metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs, model_name, save_dir='models'):
    """
    Complete training loop with validation and model saving
    
    Returns:
        Dictionary with training history
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_results = evaluate_model(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_f1'].append(val_results['f1'])
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch+1}/{num_epochs} COMPLETE - Time: {epoch_time:.2f}s")
        print(f"{'─'*60}")
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            model_path = save_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved (F1: {best_val_f1:.4f})")
        
        print(f"{'─'*60}")
    
    # Training complete summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - {model_name}")
    print(f"{'='*60}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Total Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    return history


def plot_training_history(history, title, save_path=None):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['loss', 'acc', 'f1']
    titles = ['Loss', 'Accuracy', 'F1 Score']
    
    # Plot each metric in a separate subplot
    for ax, metric, metric_title in zip(axes, metrics, titles):
        ax.plot(history[f'train_{metric}'], label='Train', marker='o')
        ax.plot(history[f'val_{metric}'], label='Validation', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_title)
        ax.set_title(f'{metric_title} - {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(labels, predictions, title, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Reject (0)', 'Accept (1)'],
                yticklabels=['Reject (0)', 'Accept (1)'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_misclassified(model, dataset, device, num_examples=10, save_path=None):
    """Visualize misclassified examples"""
    model.eval()
    misclassified = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for idx, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            
            if predicted != label:
                orig_image = dataset.get_original_image(idx)
                misclassified.append({
                    'image': orig_image,
                    'true': label.item(),
                    'pred': predicted.item()
                })
                
            if len(misclassified) >= num_examples:
                break
    
    # Plot
    n_cols = 5
    n_rows = (num_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.ravel() if num_examples > 1 else [axes]
    
    for idx, example in enumerate(misclassified[:num_examples]):
        axes[idx].imshow(example['image'])
        axes[idx].axis('off')
        axes[idx].set_title(
            f'True: {"Accept" if example["true"] == 1 else "Reject"}\n'
            f'Pred: {"Accept" if example["pred"] == 1 else "Reject"}',
            fontsize=10
        )
    
    # Hide empty subplots
    for idx in range(len(misclassified), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_experiment_results(results, experiment_name, save_dir='results'):
    """Save experiment results to JSON"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    save_path = save_dir / f"{experiment_name}_results.json"
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {save_path}")


def print_experiment_summary(results):
    """Print formatted experiment summary"""
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {results.get('model_name', 'N/A')}")
    print(f"Data Mode: {results.get('data_mode', 'N/A')}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {results.get('test_accuracy', 0):.4f}")
    print(f"  F1 Score:  {results.get('test_f1', 0):.4f}")
    print(f"  Precision: {results.get('test_precision', 0):.4f}")
    print(f"  Recall:    {results.get('test_recall', 0):.4f}")
    print(f"{'='*60}\n")
