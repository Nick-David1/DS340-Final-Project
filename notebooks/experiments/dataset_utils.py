"""
**AI-Generated** HuggingFace-based dataset utilities for furniture classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# HF dataset name
DATASET_REPO = "adybacki/furniture-profitability-dataset"


class FurnitureDataset(Dataset):
    """
    HuggingFace-based dataset for furniture classification
    """
    
    def __init__(self, hf_dataset, transforms, indices=None):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            transforms: torchvision transforms to apply
            indices: Optional list of indices to use (for train/val/test split)
        """
        self.hf_dataset = hf_dataset
        self.transforms = transforms
        self.indices = indices if indices is not None else list(range(len(hf_dataset)))
        
        print(f"Dataset initialized with {len(self.indices)} samples")
        
        # Get labels for this subset
        labels = [hf_dataset[i]['decision'] for i in self.indices]
        print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.indices[idx]
        
        # Get data from HuggingFace dataset
        item = self.hf_dataset[actual_idx]
        image = item['image']
        label = item['decision']
        
        # Apply transforms
        image = self.transforms(image)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
    def get_original_image(self, idx):
        """Get original image without transforms for visualization"""
        actual_idx = self.indices[idx]
        item = self.hf_dataset[actual_idx]
        return item['image']


def filter_hf_dataset(hf_dataset, data_mode='all'):
    """
    Apply filtering to get indices based on data mode
    
    Args:
        hf_dataset: HuggingFace dataset
        data_mode: 'all', 'high_quality', or 'brand_msrp_only'
    
    Returns:
        List of indices matching the filter
    """

    # No filtering
    if data_mode == 'all':
        return list(range(len(hf_dataset)))
    
    elif data_mode == 'high_quality':
        # High confidence in brand, MSRP, or category
        # (Filter for items with confidence >= 0.8 in any of the tabular fields)
        indices = []
        for idx in range(len(hf_dataset)):
            item = hf_dataset[idx]
            if (item['confidence_brand'] >= 0.8 or 
                item['confidence_msrp'] >= 0.8 or 
                item['confidence_category'] >= 0.8):
                indices.append(idx)
        return indices
    
    elif data_mode == 'brand_msrp_only':
        # Filter dataset for known brand and MSRP (not 'UNKNOWN' for both)
        indices = []
        for idx in range(len(hf_dataset)):
            item = hf_dataset[idx]
            if (item['brand_name'] != 'UNKNOWN' and 
                item['msrp'] != 'UNKNOWN'):
                indices.append(idx)
        return indices
    
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")


def load_and_split_hf_data(dataset_repo, data_mode='all', test_size=0.2, 
                           val_size=0.1, random_state=42, balance_classes=True):
    """
    Load HuggingFace dataset and split into train/val/test
    
    Args:
        dataset_repo: HuggingFace dataset repo
        data_mode: 'all', 'high_quality', or 'brand_msrp_only'
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        balance_classes: Whether to balance classes
    
    Returns:
        train_indices, val_indices, test_indices, hf_dataset
    """
    print(f"Loading dataset from HuggingFace: {dataset_repo}")
    
    # Load dataset
    hf_dataset = load_dataset(dataset_repo, split='train')
    print(f"Loaded {len(hf_dataset):,} total samples")
    
    # Apply filtering
    print(f"\nFiltering for data_mode: {data_mode}")
    filtered_indices = filter_hf_dataset(hf_dataset, data_mode)
    print(f"{len(filtered_indices):,} samples after filtering")
    
    # Get labels for filtered indices
    labels = [hf_dataset[i]['decision'] for i in filtered_indices]
    
    # Check class balance
    class_counts = pd.Series(labels).value_counts()
    print(f"\nOriginal class distribution: {class_counts.to_dict()}")
    
    # Balance classes to same size
    if balance_classes and len(class_counts) == 2:
        min_class_size = class_counts.min()

        # Balance classes if highly imbalanced (max/min > 2)
        if class_counts.max() / min_class_size > 2:
            print(f"Balancing classes to {min_class_size} samples each...")
            
            # Separate indices by class
            indices_0 = [idx for idx in filtered_indices if hf_dataset[idx]['decision'] == 0]
            indices_1 = [idx for idx in filtered_indices if hf_dataset[idx]['decision'] == 1]
            
            # Randomly sample to balance
            np.random.seed(random_state)
            indices_0 = np.random.choice(indices_0, min_class_size, replace=False).tolist()
            indices_1 = np.random.choice(indices_1, min_class_size, replace=False).tolist()
            
            # shuffle random samples
            filtered_indices = indices_0 + indices_1
            np.random.shuffle(filtered_indices)
            
            # Update labels after shuffling
            labels = [hf_dataset[i]['decision'] for i in filtered_indices]
            print(f"Balanced class distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    # Split into train and temp (val+test) default train size = 0.8
    train_idx, temp_idx = train_test_split(
        filtered_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Further split temp into val and test (50/50 of temp) default 0.1 each
    temp_labels = [hf_dataset[i]['decision'] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx, hf_dataset


def get_hf_data_loaders(train_idx, val_idx, test_idx, hf_dataset,
                        train_transforms, test_transforms, 
                        batch_size=32, num_workers=0):
    """
    Create DataLoaders for HuggingFace dataset
    
    Args:
        train_idx, val_idx, test_idx: Index lists for each split
        hf_dataset: HuggingFace dataset
        train_transforms: Transforms for training data
        test_transforms: Transforms for val/test data
        batch_size: Batch size
        num_workers: Num of workers
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    # Create datasets
    train_dataset = FurnitureDataset(hf_dataset, train_transforms, train_idx)
    val_dataset = FurnitureDataset(hf_dataset, test_transforms, val_idx)
    test_dataset = FurnitureDataset(hf_dataset, test_transforms, test_idx)
    
    # Create dataloaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset