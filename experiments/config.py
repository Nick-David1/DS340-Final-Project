"""
Experiment Configuration for DS340 Furniture Classification
Author: Nicholas David
Date: November 2024
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

# ==================== GLOBAL CONFIGURATION ====================

PROJECT_ROOT = "/Users/nicholasdavid/DS340 /Final Project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "models")
LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "logs")

# Create directories
for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = 4 if DEVICE.type == "cpu" else 8

# Random seed for reproducibility
RANDOM_SEED = 42

# ==================== DATA CONFIGURATION ====================

@dataclass
class DataConfig:
    """Data configuration parameters"""
    # File paths
    full_dataset_path: str = os.path.join(DATA_DIR, "DATASET_FINAL_WITH_IMAGES.csv")  # FIXED: Use full 26k dataset!
    images_dir: str = os.path.join(DATA_DIR, "images")
    
    # Dataset variants
    dataset_variants: Dict[str, str] = None
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Class labels
    class_names: List[str] = None
    num_classes: int = 2  # Accept/Reject
    
    # Preprocessing
    handle_unknown_as: str = "category"  # or "missing_indicator"
    
    def __post_init__(self):
        self.dataset_variants = {
            "all_data": "Include all items regardless of UNKNOWN values",
            "either_or_both": "Items with at least brand OR MSRP known", 
            "both_only": "Items with both brand AND MSRP known"
        }
        self.class_names = ["Reject", "Accept"]

# ==================== IMAGE MODEL CONFIGURATION ====================

@dataclass
class ImageModelConfig:
    """Configuration for CNN image models"""
    # Model architectures
    architectures: List[str] = None
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    # Data augmentation (training only)
    augmentation: Dict[str, any] = None
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    
    # Transfer learning
    use_pretrained: bool = True
    freeze_backbone_epochs: int = 5
    
    def __post_init__(self):
        self.architectures = [
            "resnet50",
            "efficientnet_b0",
            "mobilenet_v3"
        ]
        # ImageNet normalization
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.augmentation = {
            "horizontal_flip": True,
            "rotation_degrees": 15,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.1
        }

# ==================== TABULAR MODEL CONFIGURATION ====================

@dataclass
class TabularModelConfig:
    """Configuration for tabular data models"""
    # Model types
    models: Dict[str, Dict] = None
    
    # Feature engineering
    features_to_use: List[str] = None
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    ordinal_features: Dict[str, List] = None
    
    # Preprocessing
    scale_numerical: bool = True
    handle_missing: str = "indicator"  # "mean", "median", "mode", "indicator"
    
    # Training parameters
    cv_folds: int = 5
    use_class_weight: bool = True
    
    def __post_init__(self):
        self.models = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "class_weight": ["balanced"]
            },
            "decision_tree": {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "class_weight": ["balanced"]
            },
            "xgboost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "scale_pos_weight": None  # Will be calculated based on class ratio
            },
            "neural_network": {
                "hidden_sizes": [(128, 64), (256, 128, 64), (512, 256, 128)],
                "learning_rate": [1e-3, 1e-4],
                "batch_size": [32, 64],
                "epochs": [100],
                "dropout": [0.2, 0.3]
            },
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "class_weight": ["balanced"]
            }
        }
        
        self.features_to_use = [
            "brand_name",
            "msrp", 
            "condition",
            "sub_category_2_id",
            "fb_listing_price",
            "confidence_brand",
            "confidence_msrp"
        ]
        
        self.categorical_features = ["brand_name", "condition"]
        self.numerical_features = ["msrp", "fb_listing_price", "confidence_brand", "confidence_msrp"]
        self.ordinal_features = {
            "condition": ["Poor", "Fair", "Gently Used", "Excellent"],
            "sub_category_2_id": [1, 2, 3]  # Sectional, Couch, Loveseat
        }

# ==================== MULTI-MODAL CONFIGURATION ====================

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion models"""
    # Fusion strategies
    fusion_types: List[str] = None
    
    # Early fusion parameters
    early_fusion: Dict = None
    
    # Late fusion parameters
    late_fusion: Dict = None
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        self.fusion_types = ["early", "late", "attention"]
        
        self.early_fusion = {
            "cnn_feature_dim": 512,  # Output dim from CNN
            "tabular_feature_dim": 64,  # After encoding
            "fusion_hidden_dims": [256, 128, 64],
            "dropout": 0.3
        }
        
        self.late_fusion = {
            "ensemble_method": "weighted_average",  # or "voting", "stacking"
            "optimize_weights": True
        }

# ==================== EVALUATION CONFIGURATION ====================

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # Metrics to compute
    metrics: List[str] = None
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_pr_curve: bool = True
    plot_feature_importance: bool = True
    
    # Statistical tests
    perform_statistical_tests: bool = True
    significance_level: float = 0.05
    
    def __post_init__(self):
        self.metrics = [
            "accuracy",
            "precision",
            "recall", 
            "f1_score",
            "auc_roc",
            "auc_pr",
            "mcc",  # Matthews correlation coefficient
            "cohen_kappa"
        ]

# ==================== EXPERIMENT CONFIGURATION ====================

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    # Experiment settings
    experiment_name: str = "furniture_classification_comprehensive"
    run_parallel: bool = False  # Run experiments in parallel
    save_best_models: bool = True
    verbose: int = 1  # 0: silent, 1: progress, 2: detailed
    
    # Reproducibility
    seed: int = RANDOM_SEED
    deterministic: bool = True
    
    def __post_init__(self):
        # Sub-configurations
        self.data = DataConfig()
        self.image_model = ImageModelConfig()
        self.tabular_model = TabularModelConfig()
        self.multimodal = MultiModalConfig()
        self.evaluation = EvaluationConfig()

# Create global config instance
CONFIG = ExperimentConfig()
