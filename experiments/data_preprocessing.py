"""
Data Preprocessing Pipeline for Furniture Classification
Handles dataset creation, splitting, and preprocessing
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from config import CONFIG, DataConfig, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetVariantCreator:
    """Creates three dataset variants based on MSRP and Brand availability"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info(f"Initialized with dataset of shape: {self.df.shape}")
        self._analyze_missing_patterns()
    
    def _analyze_missing_patterns(self):
        """Analyze patterns of UNKNOWN values"""
        total = len(self.df)
        
        # Convert UNKNOWN to NaN for analysis
        df_analysis = self.df.copy()
        df_analysis['msrp'] = df_analysis['msrp'].replace('UNKNOWN', np.nan)
        df_analysis['brand_name'] = df_analysis['brand_name'].replace('UNKNOWN', np.nan)
        
        both_missing = df_analysis[['msrp', 'brand_name']].isna().all(axis=1).sum()
        either_missing = df_analysis[['msrp', 'brand_name']].isna().any(axis=1).sum()
        both_present = df_analysis[['msrp', 'brand_name']].notna().all(axis=1).sum()
        
        logger.info(f"Missing Pattern Analysis:")
        logger.info(f"  - Both MSRP and Brand UNKNOWN: {both_missing} ({both_missing/total*100:.1f}%)")
        logger.info(f"  - At least one UNKNOWN: {either_missing} ({either_missing/total*100:.1f}%)")
        logger.info(f"  - Both known: {both_present} ({both_present/total*100:.1f}%)")
    
    def create_variants(self) -> Dict[str, pd.DataFrame]:
        """Create three dataset variants"""
        variants = {}
        
        # Variant 1: All data (including all UNKNOWN)
        variants['all_data'] = self.df.copy()
        logger.info(f"Variant 'all_data': {len(variants['all_data'])} samples")
        
        # Variant 2: At least one of MSRP or Brand is known
        mask_either = ~((self.df['msrp'] == 'UNKNOWN') & (self.df['brand_name'] == 'UNKNOWN'))
        variants['either_or_both'] = self.df[mask_either].copy()
        logger.info(f"Variant 'either_or_both': {len(variants['either_or_both'])} samples")
        
        # Variant 3: Both MSRP and Brand are known
        mask_both = (self.df['msrp'] != 'UNKNOWN') & (self.df['brand_name'] != 'UNKNOWN')
        variants['both_only'] = self.df[mask_both].copy()
        logger.info(f"Variant 'both_only': {len(variants['both_only'])} samples")
        
        # Analyze class distribution in each variant
        for name, df_variant in variants.items():
            accept_ratio = (df_variant['decision'] == 1).mean() * 100
            logger.info(f"  {name} - Accept rate: {accept_ratio:.1f}%")
        
        return variants

class TabularPreprocessor:
    """Preprocessor for tabular features"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline"""
        # Identify feature types
        categorical_features = []
        numerical_features = []
        ordinal_features = []
        
        for col in CONFIG.tabular_model.features_to_use:
            if col not in df.columns:
                logger.warning(f"Feature {col} not in dataframe")
                continue
                
            if col in CONFIG.tabular_model.categorical_features:
                categorical_features.append(col)
            elif col in CONFIG.tabular_model.numerical_features:
                numerical_features.append(col)
            elif col == 'sub_category_2_id':
                ordinal_features.append(col)
        
        transformers = []
        
        # Numerical features pipeline
        if numerical_features:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_pipeline, numerical_features))
        
        # Categorical features pipeline
        if categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        # Ordinal features (condition)
        if 'condition' in df.columns and 'condition' in CONFIG.tabular_model.features_to_use:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Fair')),
                ('ordinal', OrdinalEncoder(categories=[CONFIG.tabular_model.ordinal_features['condition']]))
            ])
            transformers.append(('ord', ordinal_pipeline, ['condition']))
        
        # Sub-category as categorical (not ordinal)
        if 'sub_category_2_id' in df.columns:
            transformers.append(('subcat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['sub_category_2_id']))
        
        self.preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        return self.preprocessor
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data"""
        # Handle MSRP conversion
        df = df.copy()
        df['msrp'] = pd.to_numeric(df['msrp'].replace('UNKNOWN', np.nan), errors='coerce')
        df['fb_listing_price'] = pd.to_numeric(df['fb_listing_price'], errors='coerce')
        df['confidence_brand'] = pd.to_numeric(df['confidence_brand'], errors='coerce')
        df['confidence_msrp'] = pd.to_numeric(df['confidence_msrp'], errors='coerce')
        
        # Create preprocessor if not exists
        if self.preprocessor is None:
            self.create_preprocessor(df)
        
        # Transform features
        X = self.preprocessor.fit_transform(df)
        
        # Get feature names
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat' or name == 'subcat':
                encoder = transformer.named_steps['onehot'] if hasattr(transformer, 'named_steps') else transformer
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names.extend(encoder.get_feature_names_out(features))
            elif name == 'ord':
                feature_names.extend(features)
        
        self.feature_names = feature_names
        
        # Transform labels if provided
        if y is None:
            y = self.label_encoder.fit_transform(df['decision'])
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        df = df.copy()
        df['msrp'] = pd.to_numeric(df['msrp'].replace('UNKNOWN', np.nan), errors='coerce')
        df['fb_listing_price'] = pd.to_numeric(df['fb_listing_price'], errors='coerce')
        df['confidence_brand'] = pd.to_numeric(df['confidence_brand'], errors='coerce')
        df['confidence_msrp'] = pd.to_numeric(df['confidence_msrp'], errors='coerce')
        
        return self.preprocessor.transform(df)

class FurnitureImageDataset(Dataset):
    """PyTorch Dataset for furniture images"""
    
    def __init__(self, df: pd.DataFrame, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        self.images_dir = CONFIG.data.images_dir
        
    def __len__(self):
        return len(self.df)
    
    def _find_local_image(self, photo_id):
        """Find local image file by photo_id"""
        # Search in subdirectories for image matching photo_id
        for subdir in ['couch', 'sectional', 'loveseat', 'other']:
            subdir_path = os.path.join(self.images_dir, subdir)
            if os.path.exists(subdir_path):
                # Look for files starting with photo_id
                for filename in os.listdir(subdir_path):
                    if filename.startswith(str(photo_id) + '_'):
                        return os.path.join(subdir_path, filename)
        return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Try to find local image first using photo_id
        photo_id = row['photo_id']
        image_path = self._find_local_image(photo_id)
        
        # Fallback to URL path if local not found
        if image_path is None:
            image_path = row['photo']
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.images_dir, os.path.basename(image_path))
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Only log error once per 100 images to avoid spam
            if idx % 100 == 0:
                logger.debug(f"Could not load image for photo_id {photo_id}: {e}")
            # Create a blank image as fallback
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = int(row['decision'])
        
        # Get additional features for multi-modal
        features = {
            'image': image,
            'label': label,
            'photo_id': row['photo_id'],
            'tabular_idx': idx  # Index for matching with tabular features
        }
        
        return features

class DataModule:
    """Main data module handling all data operations"""
    
    def __init__(self, variant_name: str = 'all_data'):
        self.variant_name = variant_name
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.tabular_preprocessor = TabularPreprocessor(CONFIG.data)
        
    def setup(self):
        """Load and prepare data"""
        # Load full dataset
        logger.info(f"Loading dataset from {CONFIG.data.full_dataset_path}")
        full_df = pd.read_csv(CONFIG.data.full_dataset_path)
        
        # Create variants
        creator = DatasetVariantCreator(full_df)
        variants = creator.create_variants()
        
        # Select variant
        self.df = variants[self.variant_name]
        logger.info(f"Using variant '{self.variant_name}' with {len(self.df)} samples")
        
        # Split data
        self._create_splits()
        
    def _create_splits(self):
        """Create train/val/test splits"""
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=CONFIG.data.test_ratio,
            stratify=self.df['decision'],
            random_state=CONFIG.seed
        )
        
        # Second split: train vs val
        val_ratio_adjusted = CONFIG.data.val_ratio / (1 - CONFIG.data.test_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df['decision'],
            random_state=CONFIG.seed
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        logger.info(f"Data splits created:")
        logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(self.df)*100:.1f}%)")
        logger.info(f"  Val: {len(val_df)} ({len(val_df)/len(self.df)*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} ({len(test_df)/len(self.df)*100:.1f}%)")
        
        # Log class distributions
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            accept_rate = (df['decision'] == 1).mean() * 100
            logger.info(f"  {name} accept rate: {accept_rate:.1f}%")
    
    def get_tabular_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get preprocessed tabular data"""
        # Fit preprocessor on train data
        X_train, y_train = self.tabular_preprocessor.fit_transform(self.train_df)
        
        # Transform val and test
        X_val = self.tabular_preprocessor.transform(self.val_df)
        y_val = self.tabular_preprocessor.label_encoder.transform(self.val_df['decision'])
        
        X_test = self.tabular_preprocessor.transform(self.test_df)
        y_test = self.tabular_preprocessor.label_encoder.transform(self.test_df['decision'])
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'feature_names': self.tabular_preprocessor.feature_names
        }
    
    def get_image_datasets(self, train_transform=None, val_transform=None) -> Dict[str, Dataset]:
        """Get PyTorch image datasets"""
        datasets = {
            'train': FurnitureImageDataset(self.train_df, transform=train_transform, mode='train'),
            'val': FurnitureImageDataset(self.val_df, transform=val_transform, mode='val'),
            'test': FurnitureImageDataset(self.test_df, transform=val_transform, mode='test')
        }
        return datasets
    
    def get_multimodal_data(self) -> Dict:
        """Get both tabular and image data for multi-modal models"""
        # Get tabular data
        tabular_data = self.get_tabular_data()
        
        # Get image datasets
        image_transforms = self._get_image_transforms()
        image_datasets = self.get_image_datasets(
            train_transform=image_transforms['train'],
            val_transform=image_transforms['val']
        )
        
        return {
            'tabular': tabular_data,
            'image_datasets': image_datasets,
            'train_df': self.train_df,
            'val_df': self.val_df,
            'test_df': self.test_df
        }
    
    def _get_image_transforms(self) -> Dict[str, transforms.Compose]:
        """Get image transforms for training and validation"""
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(CONFIG.image_model.image_size[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(CONFIG.image_model.augmentation['rotation_degrees']),
            transforms.ColorJitter(
                brightness=CONFIG.image_model.augmentation['brightness'],
                contrast=CONFIG.image_model.augmentation['contrast'],
                saturation=CONFIG.image_model.augmentation['saturation']
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CONFIG.image_model.normalize_mean,
                std=CONFIG.image_model.normalize_std
            )
        ])
        
        # Validation/test transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(CONFIG.image_model.image_size),
            transforms.CenterCrop(CONFIG.image_model.image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CONFIG.image_model.normalize_mean,
                std=CONFIG.image_model.normalize_std
            )
        ])
        
        return {'train': train_transform, 'val': val_transform}
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        class_counts = np.bincount(self.train_df['decision'])
        class_weights = len(self.train_df) / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)

if __name__ == "__main__":
    # Test the preprocessing pipeline
    logger.info("Testing data preprocessing pipeline...")
    
    # Test each variant
    for variant_name in ['all_data', 'either_or_both', 'both_only']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing variant: {variant_name}")
        logger.info(f"{'='*50}")
        
        data_module = DataModule(variant_name=variant_name)
        data_module.setup()
        
        # Test tabular data
        tabular_data = data_module.get_tabular_data()
        logger.info(f"Tabular data shapes:")
        logger.info(f"  Train: {tabular_data['train'][0].shape}")
        logger.info(f"  Val: {tabular_data['val'][0].shape}")
        logger.info(f"  Test: {tabular_data['test'][0].shape}")
        logger.info(f"  Features: {len(tabular_data['feature_names'])}")
        
        # Test class weights
        weights = data_module.get_class_weights()
        logger.info(f"Class weights: {weights}")
    
    logger.info("\nData preprocessing pipeline test complete!")
