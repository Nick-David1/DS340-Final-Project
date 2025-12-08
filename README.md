# DS340 Final Project: Furniture Saleability Prediction



---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Quick Start](#quick-start)
4. [Repository Structure](#repository-structure)
5. [Reproduction Instructions](#reproduction-instructions)
6. [Dataset Information](#dataset-information)
7. [AI-Powered Data Labeling](#ai-powered-data-labeling)
8. [Experimental Design](#experimental-design)
9. [Technology Stack](#technology-stack)
10. [Paper & Figures](#paper--figures)

---

##  Project Overview

This project develops a **machine learning system to predict furniture saleability** for consignment businesses. Given a furniture listing (images, title, description, and metadata), the system predicts whether to **Accept** (estimated resale value > $500) or **Reject** the item for consignment pickup.

### Business Context

second-edition.co, a consignment furniture business, receives 100+ daily submissions. Manual review is time-consuming (2 min/item) and inconsistent. This system automates 85% of decisions, saving 1,017 hours/year and $30,510 in labor costs.

### Approach

We experimented with three model types:
1. **Image-only models** (CNNs: ResNet50, EfficientNet-B0, MobileNet-V3, DenseNet121)
2. **Tabular-only models** (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, Neural Networks)
3. **Multi-modal fusion models** (Early fusion, Late fusion with attention)

### Key Finding

**Tabular models vastly outperformed multi-modal and image-only approaches.** Stacking Ensemble (tabular) achieved **85.0% accuracy and 93.0% AUC**, while multi-modal models achieved only 61% F1 score. This reveals that **structured features (brand, MSRP, condition) are far more predictive than visual features** for furniture resale value prediction.

---

##  Key Results

### Best Model: Stacking Ensemble
- **Accuracy**: 85.0%
- **F1 Score**: 85.2%
- **AUC-ROC**: 93.0%
- **Precision (Accept)**: 83.4%
- **Recall (Accept)**: 85.5%

### Model Comparison

| Model | Type | Accuracy | F1 | AUC |
|-------|------|----------|-----|-----|
| **Stacking Ensemble** | Tabular | **85.0%** | **85.2%** | **93.0%** |
| LightGBM | Tabular | 84.7% | 84.9% | 92.5% |
| CatBoost | Tabular | 84.5% | 84.7% | 93.0% |
| XGBoost | Tabular | 76.6% | 76.3% | 83.7% |
| Random Forest | Tabular | 76.0% | 75.7% | 83.2% |
| Early Fusion | Multi-Modal | 65.0% | 65.0% | 72.0% |
| EfficientNet-B0 | Image | 61.7% | 61.7% | 63.9% |

### Performance by Model Type
- **Tabular**: 74.1% avg F1, 80.2% avg AUC 
- **Multi-Modal**: 61.1% avg F1, 66.3% avg AUC
- **Image**: 60.4% avg F1, 63.8% avg AUC

**Key Insight**: Tabular models outperform multi-modal by 13 percentage points, challenging the assumption that "more modalities = better performance."

---

##  Quick Start

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM recommended
- GPU optional (CPU training works but slower for image models)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd "DS340 /Final Project"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python3 -c "import torch; import sklearn; import xgboost; print(' All dependencies installed')"
```

### Quick Test (5 minutes)

```bash
# Run experiments on sample data
cd experiments
python3 run_experiments.py --variant both_only --models advanced_models

# View results
ls -lh results/advanced_models_both_only.json
```

---

##  Repository Structure

```
DS340 Final Project/
│
├── data/                                    # Datasets and images
│   ├── DATASET_FINAL_WITH_IMAGES.csv       #  Main dataset (26,376 items)
│   ├── sample_data.csv                     # Small subset for testing
│   └── images/                             # Local furniture images (26K+)
│       ├── sectional/  (14,441 images)
│       ├── couch/      (8,604 images)
│       └── loveseat/   (3,331 images)
│
├── src/                                     # AI data labeling system
│   ├── production_furniture_labeler.py     # Main labeling system
│   └── parallel_furniture_labeler.py       # High-speed parallel version
│
├── experiments/                             # ML experiments
│   ├── config.py                           # Configuration settings
│   ├── data_preprocessing.py               # Data loading & splitting
│   ├── tabular_models.py                   # Tabular experiments
│   ├── image_models.py                     # CNN experiments
│   ├── multimodal_models.py                # Fusion experiments
│   ├── advanced_models.py                  #  Ensemble & advanced techniques
│   ├── run_experiments.py                  #  Main entry point
│   ├── generate_paper_figures.py           # Paper visualization generation
│   │
│   ├── models/                             # Trained model files
│   │   ├── stacking_both_only.pkl          #  Best model
│   │   ├── lightgbm_both_only.pkl
│   │   ├── catboost_both_only.cbm
│   │   └── ... (all experimental models)
│   │
│   ├── results/                            # Experimental results
│   │   ├── advanced_models_both_only.json  #  Final results
│   │   └── exp_*/                          # Individual experiment runs
│   │
│   └── paper_figures/                      # Publication-ready figures
│       ├── fig1_model_comparison.png
│       ├── fig2_confusion_matrix.png
│       ├── fig3_modality_comparison.png
│       ├── fig4_stacking_architecture.png
│       ├── fig5_data_quality_impact.png
│       └── fig6_training_curves.png
│
├── README.md                                #  This file
├── RESULTS_SUMMARY.md                       # Comprehensive results summary
├── PRESENTATION_OUTLINE.md                  # Lightning talk outline
├── requirements.txt                         # Python dependencies
└── .gitignore                              # Git exclusions

```

---

##  Reproduction Instructions

### Step 1: Environment Setup (5 minutes)

```bash
# Activate virtual environment
source venv/bin/activate

# Verify GPU availability (optional)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Data Verification (2 minutes)

```bash
# Verify dataset exists
ls -lh data/DATASET_FINAL_WITH_IMAGES.csv

# Check dataset statistics
python3 -c "
import pandas as pd
df = pd.read_csv('data/DATASET_FINAL_WITH_IMAGES.csv')
print(f'Total samples: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Accept rate: {(df[\"decision\"]==1).mean()*100:.1f}%')
"
```

**Expected output**:
```
Total samples: 26376
Columns: ['photo_id', 'photo', 'condition', 'sub_category_2_id', 'title', 'msrp', 'brand_name', ...]
Accept rate: 28.5%
```

### Step 3: Run Experiments (30-60 minutes)

#### Option A: Run All Experiments (Full Reproduction)

```bash
cd experiments

# Run all model types on all dataset variants
python3 run_experiments.py \
  --variants both_only either_or_both all_data \
  --models tabular image multimodal advanced

# This will:
# - Train ~20 different models
# - Generate all results files
# - Save trained models
# - Take 30-60 minutes depending on hardware
```

#### Option B: Run Best Model Only (Quick Reproduction)

```bash
cd experiments

# Train only the stacking ensemble (best model)
python3 advanced_models.py --variants both_only

# Expected output:
#  Stacking Ensemble: 85.0% Acc, 85.2% F1, 93.0% AUC
# Model saved to: models/stacking_both_only.pkl
```

#### Option C: Use Pre-trained Models (Fastest)

```bash
cd experiments

# Load and evaluate existing models
python3 -c "
import pickle
with open('models/stacking_both_only.pkl', 'rb') as f:
    model = pickle.load(f)
print(' Model loaded successfully')
print(f'Model type: {type(model).__name__}')
"
```

### Step 4: Generate Visualizations (5 minutes)

```bash
cd experiments

# Generate all paper figures
python3 generate_paper_figures.py

# Outputs saved to: experiments/paper_figures/
ls -lh paper_figures/
```

### Step 5: View Results

```bash
# View comprehensive results
cat experiments/results/advanced_models_both_only.json | python3 -m json.tool

# View summary
cat ../RESULTS_SUMMARY.md
```

---

##  Dataset Information

### Source
- **Origin**: Facebook Marketplace furniture listings (Boston area)
- **Collection period**: October-November 2025
- **Total items**: 26,376 furniture listings
- **Categories**: Sectionals (54.7%), Couches (32.6%), Loveseats (12.6%)

### Labeling Method
**AI-powered labeling using Google Gemini 2.5 Flash** (see next section for details)

### Dataset Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `photo_id` | int | Unique item identifier | 1156725039975668 |
| `photo` | str | CDN image URL | https://... |
| `local_image_path` | str | Local image file path | images/couch/... |
| `condition` | str | Item condition | "Gently Used" |
| `sub_category_2_id` | int | Category (1=Sectional, 2=Couch, 3=Loveseat) | 2 |
| `title` | str | AI-extracted title | "Ashley Furniture Sofa" |
| `msrp` | float/str | Original retail price | 1200 or "UNKNOWN" |
| `brand_name` | str | Brand name | "Ashley Furniture" |
| `confidence_brand` | float | Brand confidence score | 0.95 |
| `confidence_msrp` | float | MSRP confidence score | 0.85 |
| `msrp_source` | str | MSRP data source | "description" / "web_search" |
| `reasoning` | str | AI labeling reasoning | "High-end brand..." |
| `red_flags` | str | Quality concerns | "pet home, stains" |
| `decision` | int | **Target variable** (1=Accept, 0=Reject) | 1 |
| `listing_title` | str | Original Facebook title | "Grey Fluffy Couch" |
| `listing_description` | str | Full listing description | "Couch is super soft..." |
| `fb_attribute_condition` | json | Facebook condition data | {...} |
| `fb_listing_price` | float | Asking price on Facebook | 100 |
| `fb_is_sold` | bool | Whether item sold | FALSE |

### Data Splits

**`both_only` variant** (used for best results):
- Total: 2,416 samples (items with both brand AND MSRP known)
- Train: 1,690 samples (70%)
- Validation: 363 samples (15%)
- Test: 363 samples (15%)
- **Class distribution**: 55% Accept, 45% Reject
- **Balancing**: SMOTE applied to training set only (50/50), validation/test kept imbalanced

**Random seed**: 42 (ensures reproducibility)

---

##  AI-Powered Data Labeling

### The Challenge

**Cold Start Problem**: No labeled training data existed. Traditional manual labeling would require:
- 100+ hours of work
- Domain expertise (furniture brands, pricing)
- $3,000+ cost at $30/hr

### Our Solution

Built a **AI labeling system** using Google Gemini 2.5 Flash multimodal API:

**System Architecture**:
1. **Stage 1**: Filter non-seating furniture (separate LLM call)
2. **Stage 2**: Extract brand name (LLM + title analysis)
3. **Stage 3**: Extract MSRP (LLM + Google Search in parallel)
4. **Stage 4**: Verify with domain expert decision logic 

**Multi-Source Validation**:
- Image analysis (visual condition, brand logos)
- Title parsing (brand extraction)
- Description mining (MSRP mentions, condition details)
- Web search (MSRP validation via Google)
- Weighted confidence scoring (Title 40%, Desc 35%, Image 20%, Web 5%)

**Performance**:
- **Labeled 26,376 items in ~25 minutes**
- **500 concurrent workers** (maxing out Gemini API rate limits)
- 90-95% brand accuracy
- 85-90% MSRP accuracy (within ±20% acceptable range)
- Fully reproducible and scalable

**Code**: See `src/production_furniture_labeler.py` and `src/parallel_furniture_labeler.py`


---

##  Experimental Design

### Research Questions

1. **RQ1**: Do multi-modal models outperform single-modality models for furniture saleability prediction?
2. **RQ2**: Which features are most predictive: visual (images) or structured (brand, MSRP, condition)?
3. **RQ3**: How does data quality (completeness of brand/MSRP) impact model performance?
4. **RQ4**: Can ensemble methods improve upon single model performance?

### Models Tested

**Tabular Models**:
- XGBoost, LightGBM, CatBoost
- Random Forest, Gradient Boosting, Extra Trees, Decision Trees
- Logistic Regression, Neural Networks
- **Stacking Ensemble** (meta-learner combining CatBoost, LightGBM, RF, GB)

**Image Models**:
- ResNet50, EfficientNet-B0, MobileNet-V3, DenseNet121
- Transfer learning from ImageNet
- Data augmentation: flips, rotations, color jitter

**Multi-Modal Models**:
- Early Fusion (concatenate features → classifier)
- Late Fusion Weighted (ensemble predictions)
- Late Fusion Attention (learned attention weights)

### Evaluation Methodology

- **Primary metric**: F1 Score (weighted)
- **Secondary metrics**: Accuracy, Precision, Recall, AUC-ROC
- **Cross-validation**: 5-fold stratified CV for robustness
- **Class balancing**: SMOTE oversampling on training set only
- **Hyperparameter tuning**: Grid search with CV

### Key Findings

1.  **Tabular >> Multi-Modal** (85% vs 61% F1): Structured features dominate
2.  **Quality > Quantity** (2.4K vs 26K samples): Complete data beats large incomplete data by 24%
3.  **Stacking Ensemble wins** (85% vs 76% XGBoost): Combining models captures complementary strengths
4.  **Multi-Modal Paradox**: Weak image features dragged down strong tabular features
5.  **Feature importance**: MSRP (23.4%) + Brand (18.7%) = 42% of predictive power; images ~3%

---

## ️ Technology Stack

### Core ML/DL Libraries
- **PyTorch** 2.0+ (deep learning framework)
- **scikit-learn** 1.3+ (traditional ML, preprocessing, evaluation)
- **XGBoost** 2.0+ (gradient boosting)
- **LightGBM** 4.0+ (fast gradient boosting)
- **CatBoost** 1.2+ (categorical boosting)
- **imbalanced-learn** (SMOTE for class balancing)

### Data Processing
- **pandas** 2.0+ (data manipulation)
- **NumPy** 1.24+ (numerical computing)
- **Pillow** 10.0+ (image processing)

### Visualization
- **Matplotlib** 3.7+ (plotting)
- **seaborn** 0.12+ (statistical visualization)

### AI Labeling (Optional)
- **google-genai** (Gemini API) - Only needed for data labeling, not for experiments

### Development Tools
- Python 3.10+
- Git for version control
- Virtual environment (venv)

---

##  Paper & Figures

### Figures

All figures are in `experiments/paper_figures/` (both PNG at 300 DPI and PDF):

1. **fig1_model_comparison.png**: Performance comparison across all models
2. **fig2_confusion_matrix.png**: Stacking Ensemble confusion matrix
3. **fig3_modality_comparison.png**: Multi-modal paradox visualization
4. **fig4_stacking_architecture.png**: Ensemble architecture diagram
5. **fig5_data_quality_impact.png**: Quality vs. quantity trade-off
6. **fig6_training_curves.png**: Loss and F1 score progression





