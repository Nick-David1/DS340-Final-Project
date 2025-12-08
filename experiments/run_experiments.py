#!/usr/bin/env python3
"""
Main Experiment Runner for DS340 Furniture Classification Project
Orchestrates all experiments across dataset variants and model types
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, RESULTS_DIR, LOGS_DIR
from data_preprocessing import DataModule, DatasetVariantCreator
from image_models import run_image_experiments
from tabular_models import run_tabular_experiments
from multimodal_models import run_multimodal_experiments

# Setup logging
log_file = os.path.join(LOGS_DIR, f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment orchestrator"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join(RESULTS_DIR, self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.all_results = {}
        
    def run_dataset_variant_experiments(self, variant_name: str, 
                                       run_image: bool = True,
                                       run_tabular: bool = True,
                                       run_multimodal: bool = True) -> Dict:
        """Run all experiments for a specific dataset variant"""
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING EXPERIMENTS FOR DATASET VARIANT: {variant_name}")
        logger.info(f"{'='*80}")
        
        # Initialize data module
        data_module = DataModule(variant_name=variant_name)
        data_module.setup()
        
        variant_results = {
            'variant_name': variant_name,
            'dataset_stats': self._get_dataset_stats(data_module)
        }
        
        # Run image experiments
        if run_image:
            logger.info(f"\n--- Running Image-Only Experiments ---")
            try:
                image_results = run_image_experiments(data_module, variant_name)
                variant_results['image_models'] = image_results
                self._save_results(image_results, f"{variant_name}_image_results.json")
            except Exception as e:
                logger.error(f"Error in image experiments: {e}")
                variant_results['image_models'] = {'error': str(e)}
        
        # Run tabular experiments
        if run_tabular:
            logger.info(f"\n--- Running Tabular-Only Experiments ---")
            try:
                tabular_results = run_tabular_experiments(data_module, variant_name)
                variant_results['tabular_models'] = tabular_results
                self._save_results(tabular_results, f"{variant_name}_tabular_results.json")
            except Exception as e:
                logger.error(f"Error in tabular experiments: {e}")
                variant_results['tabular_models'] = {'error': str(e)}
        
        # Run multi-modal experiments
        if run_multimodal:
            logger.info(f"\n--- Running Multi-Modal Experiments ---")
            try:
                multimodal_results = run_multimodal_experiments(data_module, variant_name)
                variant_results['multimodal_models'] = multimodal_results
                self._save_results(multimodal_results, f"{variant_name}_multimodal_results.json")
            except Exception as e:
                logger.error(f"Error in multi-modal experiments: {e}")
                variant_results['multimodal_models'] = {'error': str(e)}
        
        return variant_results
    
    def run_all_experiments(self, variants: Optional[List[str]] = None,
                           run_image: bool = True,
                           run_tabular: bool = True,
                           run_multimodal: bool = True):
        """Run experiments for all dataset variants"""
        if variants is None:
            variants = ['all_data', 'either_or_both', 'both_only']
        
        for variant_name in variants:
            variant_results = self.run_dataset_variant_experiments(
                variant_name, 
                run_image=run_image,
                run_tabular=run_tabular,
                run_multimodal=run_multimodal
            )
            self.all_results[variant_name] = variant_results
        
        # Save comprehensive results
        self._save_all_results()
        
        # Generate comparison report
        self.generate_comparison_report()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ALL EXPERIMENTS COMPLETED!")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"{'='*80}")
    
    def _get_dataset_stats(self, data_module) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(data_module.df),
            'train_samples': len(data_module.train_df),
            'val_samples': len(data_module.val_df),
            'test_samples': len(data_module.test_df),
            'accept_rate_total': (data_module.df['decision'] == 1).mean(),
            'accept_rate_train': (data_module.train_df['decision'] == 1).mean(),
            'accept_rate_val': (data_module.val_df['decision'] == 1).mean(),
            'accept_rate_test': (data_module.test_df['decision'] == 1).mean(),
            'unknown_msrp_rate': (data_module.df['msrp'] == 'UNKNOWN').mean(),
            'unknown_brand_rate': (data_module.df['brand_name'] == 'UNKNOWN').mean()
        }
        return stats
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to file"""
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy arrays and non-serializable objects
        def convert_results(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_results(item) for item in obj]
            return obj
        
        clean_results = convert_results(results)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _save_all_results(self):
        """Save all experiment results"""
        # Save as JSON
        self._save_results(self.all_results, "all_experiment_results.json")
        
        # Save as pickle for complete preservation
        pickle_path = os.path.join(self.results_dir, "all_experiment_results.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.all_results, f)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report...")
        
        # Create results dataframe
        results_data = []
        
        for variant_name, variant_results in self.all_results.items():
            # Process image results
            if 'image_models' in variant_results and 'error' not in variant_results['image_models']:
                for model_name, model_results in variant_results['image_models'].items():
                    if 'test_results' in model_results:
                        results_data.append({
                            'variant': variant_name,
                            'model_type': 'Image',
                            'model_name': model_name,
                            'accuracy': model_results['test_results']['accuracy'],
                            'f1_weighted': model_results['test_results']['f1_weighted'],
                            'f1_macro': model_results['test_results']['f1_macro'],
                            'auc': model_results['test_results']['auc'],
                            'precision_weighted': model_results['test_results']['precision_weighted'],
                            'recall_weighted': model_results['test_results']['recall_weighted']
                        })
            
            # Process tabular results
            if 'tabular_models' in variant_results and 'error' not in variant_results['tabular_models']:
                for model_name, model_results in variant_results['tabular_models'].items():
                    if 'test_results' in model_results:
                        results_data.append({
                            'variant': variant_name,
                            'model_type': 'Tabular',
                            'model_name': model_name,
                            'accuracy': model_results['test_results']['accuracy'],
                            'f1_weighted': model_results['test_results']['f1_weighted'],
                            'f1_macro': model_results['test_results']['f1_macro'],
                            'auc': model_results['test_results']['auc'],
                            'precision_weighted': model_results['test_results']['precision_weighted'],
                            'recall_weighted': model_results['test_results']['recall_weighted']
                        })
            
            # Process multi-modal results
            if 'multimodal_models' in variant_results and 'error' not in variant_results['multimodal_models']:
                for model_name, model_results in variant_results['multimodal_models'].items():
                    if 'test_results' in model_results:
                        results_data.append({
                            'variant': variant_name,
                            'model_type': 'Multi-Modal',
                            'model_name': model_name,
                            'accuracy': model_results['test_results']['accuracy'],
                            'f1_weighted': model_results['test_results']['f1_weighted'],
                            'f1_macro': model_results['test_results']['f1_macro'],
                            'auc': model_results['test_results']['auc'],
                            'precision_weighted': model_results['test_results']['precision_weighted'],
                            'recall_weighted': model_results['test_results']['recall_weighted']
                        })
        
        # Create DataFrame
        if results_data:
            results_df = pd.DataFrame(results_data)
            
            # Save to CSV
            csv_path = os.path.join(self.results_dir, "results_comparison.csv")
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Results comparison saved to {csv_path}")
            
            # Generate visualizations
            self._create_visualizations(results_df)
            
            # Generate text report
            self._generate_text_report(results_df)
        else:
            logger.warning("No results to compare")
    
    def _create_visualizations(self, results_df: pd.DataFrame):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. F1 Score Comparison across variants
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, variant in enumerate(['all_data', 'either_or_both', 'both_only']):
            if variant in results_df['variant'].values:
                variant_df = results_df[results_df['variant'] == variant]
                
                # Group by model type
                ax = axes[idx]
                variant_df_pivot = variant_df.pivot_table(
                    values='f1_weighted',
                    index='model_name',
                    columns='model_type',
                    aggfunc='mean'
                )
                
                variant_df_pivot.plot(kind='bar', ax=ax, rot=45)
                ax.set_title(f'F1 Score - {variant}')
                ax.set_xlabel('Model')
                ax.set_ylabel('F1 Score (Weighted)')
                ax.legend(title='Model Type')
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'f1_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Best Models Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get best model per variant and type
        best_models = results_df.loc[results_df.groupby(['variant', 'model_type'])['f1_weighted'].idxmax()]
        
        # Pivot for plotting
        best_pivot = best_models.pivot(index='variant', columns='model_type', values='f1_weighted')
        best_pivot.plot(kind='bar', ax=ax)
        
        ax.set_title('Best F1 Scores by Dataset Variant and Model Type')
        ax.set_xlabel('Dataset Variant')
        ax.set_ylabel('F1 Score (Weighted)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(title='Model Type')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'best_models_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. AUC-ROC Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter for models with AUC scores
        auc_df = results_df[results_df['auc'] > 0]
        
        if not auc_df.empty:
            auc_pivot = auc_df.pivot_table(
                values='auc',
                index='model_name',
                columns='variant',
                aggfunc='mean'
            )
            
            auc_pivot.plot(kind='bar', ax=ax)
            ax.set_title('AUC-ROC Scores Across Models and Variants')
            ax.set_xlabel('Model')
            ax.set_ylabel('AUC-ROC')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title='Dataset Variant')
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'auc_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Model Type Performance Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap data
        heatmap_data = results_df.pivot_table(
            values='f1_weighted',
            index='model_name',
            columns='variant',
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'F1 Score (Weighted)'})
        ax.set_title('F1 Score Heatmap: Models vs Dataset Variants')
        ax.set_xlabel('Dataset Variant')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'f1_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def _generate_text_report(self, results_df: pd.DataFrame):
        """Generate detailed text report"""
        report_path = os.path.join(self.results_dir, "experiment_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DS340 FURNITURE CLASSIFICATION - EXPERIMENT REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Dataset Statistics
            f.write("DATASET STATISTICS\n")
            f.write("-"*40 + "\n")
            for variant_name, variant_results in self.all_results.items():
                if 'dataset_stats' in variant_results:
                    stats = variant_results['dataset_stats']
                    f.write(f"\n{variant_name}:\n")
                    f.write(f"  Total samples: {stats['total_samples']}\n")
                    f.write(f"  Train/Val/Test: {stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}\n")
                    f.write(f"  Accept rate: {stats['accept_rate_total']:.2%}\n")
                    f.write(f"  Unknown MSRP: {stats['unknown_msrp_rate']:.2%}\n")
                    f.write(f"  Unknown Brand: {stats['unknown_brand_rate']:.2%}\n")
            
            # Best Performing Models
            f.write("\n" + "="*80 + "\n")
            f.write("BEST PERFORMING MODELS\n")
            f.write("-"*40 + "\n")
            
            # Find best overall
            best_overall = results_df.loc[results_df['f1_weighted'].idxmax()]
            f.write(f"\nBest Overall Model:\n")
            f.write(f"  Model: {best_overall['model_name']}\n")
            f.write(f"  Type: {best_overall['model_type']}\n")
            f.write(f"  Variant: {best_overall['variant']}\n")
            f.write(f"  F1 Score: {best_overall['f1_weighted']:.4f}\n")
            f.write(f"  AUC: {best_overall['auc']:.4f}\n")
            f.write(f"  Accuracy: {best_overall['accuracy']:.4f}\n")
            
            # Best per model type
            f.write("\nBest by Model Type:\n")
            for model_type in results_df['model_type'].unique():
                type_df = results_df[results_df['model_type'] == model_type]
                if not type_df.empty:
                    best = type_df.loc[type_df['f1_weighted'].idxmax()]
                    f.write(f"\n  {model_type}:\n")
                    f.write(f"    Model: {best['model_name']}\n")
                    f.write(f"    Variant: {best['variant']}\n")
                    f.write(f"    F1 Score: {best['f1_weighted']:.4f}\n")
                    f.write(f"    AUC: {best['auc']:.4f}\n")
            
            # Best per variant
            f.write("\nBest by Dataset Variant:\n")
            for variant in results_df['variant'].unique():
                variant_df = results_df[results_df['variant'] == variant]
                if not variant_df.empty:
                    best = variant_df.loc[variant_df['f1_weighted'].idxmax()]
                    f.write(f"\n  {variant}:\n")
                    f.write(f"    Model: {best['model_name']}\n")
                    f.write(f"    Type: {best['model_type']}\n")
                    f.write(f"    F1 Score: {best['f1_weighted']:.4f}\n")
                    f.write(f"    AUC: {best['auc']:.4f}\n")
            
            # Key Findings
            f.write("\n" + "="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("-"*40 + "\n")
            
            # Compare model types
            avg_by_type = results_df.groupby('model_type')['f1_weighted'].mean()
            f.write("\nAverage F1 Score by Model Type:\n")
            for model_type, avg_score in avg_by_type.items():
                f.write(f"  {model_type}: {avg_score:.4f}\n")
            
            # Compare variants
            avg_by_variant = results_df.groupby('variant')['f1_weighted'].mean()
            f.write("\nAverage F1 Score by Dataset Variant:\n")
            for variant, avg_score in avg_by_variant.items():
                f.write(f"  {variant}: {avg_score:.4f}\n")
            
            # Multi-modal vs single-modal
            if 'Multi-Modal' in results_df['model_type'].values:
                multimodal_avg = results_df[results_df['model_type'] == 'Multi-Modal']['f1_weighted'].mean()
                image_avg = results_df[results_df['model_type'] == 'Image']['f1_weighted'].mean()
                tabular_avg = results_df[results_df['model_type'] == 'Tabular']['f1_weighted'].mean()
                
                f.write("\nMulti-Modal vs Single-Modal Performance:\n")
                f.write(f"  Multi-Modal average: {multimodal_avg:.4f}\n")
                f.write(f"  Image-only average: {image_avg:.4f}\n")
                f.write(f"  Tabular-only average: {tabular_avg:.4f}\n")
                
                improvement_image = (multimodal_avg - image_avg) / image_avg * 100
                improvement_tabular = (multimodal_avg - tabular_avg) / tabular_avg * 100
                f.write(f"  Improvement over image-only: {improvement_image:+.1f}%\n")
                f.write(f"  Improvement over tabular-only: {improvement_tabular:+.1f}%\n")
            
            # Impact of unknown values
            if len(results_df['variant'].unique()) >= 3:
                all_data_avg = results_df[results_df['variant'] == 'all_data']['f1_weighted'].mean()
                either_avg = results_df[results_df['variant'] == 'either_or_both']['f1_weighted'].mean()
                both_avg = results_df[results_df['variant'] == 'both_only']['f1_weighted'].mean()
                
                f.write("\nImpact of Unknown Values:\n")
                f.write(f"  All data (including unknowns): {all_data_avg:.4f}\n")
                f.write(f"  At least one known: {either_avg:.4f}\n")
                f.write(f"  Both known: {both_avg:.4f}\n")
                f.write(f"  Performance drop with unknowns: {(both_avg - all_data_avg)/both_avg*100:.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Text report saved to {report_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run DS340 Furniture Classification Experiments')
    parser.add_argument('--variants', nargs='+', default=['all_data', 'either_or_both', 'both_only'],
                       help='Dataset variants to test')
    parser.add_argument('--no-image', action='store_true', help='Skip image experiments')
    parser.add_argument('--no-tabular', action='store_true', help='Skip tabular experiments')
    parser.add_argument('--no-multimodal', action='store_true', help='Skip multi-modal experiments')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--quick', action='store_true', help='Run quick test with reduced epochs')
    
    args = parser.parse_args()
    
    # Adjust config for quick test
    if args.quick:
        logger.info("Running in quick test mode with reduced epochs")
        CONFIG.image_model.epochs = 2
        CONFIG.multimodal.epochs = 2
        CONFIG.tabular_model.models['neural_network']['epochs'] = [10]
    
    # Create experiment runner
    runner = ExperimentRunner(experiment_name=args.name)
    
    # Run experiments
    runner.run_all_experiments(
        variants=args.variants,
        run_image=not args.no_image,
        run_tabular=not args.no_tabular,
        run_multimodal=not args.no_multimodal
    )
    
    logger.info("All experiments completed successfully!")

if __name__ == "__main__":
    main()
