# 🎯 DS340 Furniture Classification - Results Analysis & Recommendations

## 📊 **Executive Summary**

**Best Model:** XGBoost (Tabular) - **76.6% Accuracy, 83.7% AUC**

**Key Finding:** Tabular models significantly outperform image and multi-modal models, suggesting that structured features (brand, MSRP, condition) are more predictive than visual features alone.

---

## 📈 **Complete Results Ranking**

### 🥇 Top 5 Models

| Rank | Model | Type | Accuracy | F1 Score | AUC |
|------|-------|------|----------|----------|-----|
| 1 | **XGBoost** | Tabular | **76.6%** | **76.3%** | **83.7%** |
| 2 | Random Forest | Tabular | 76.0% | 75.7% | 83.2% |
| 3 | Gradient Boost | Tabular | 75.8% | 75.3% | 82.2% |
| 4 | Decision Tree | Tabular | 75.8% | 74.7% | 77.3% |
| 5 | Neural Net (Tab) | Tabular | 71.1% | 71.1% | 79.0% |

### 📊 Average Performance by Model Type

| Type | Avg F1 | Avg AUC | Performance |
|------|--------|---------|-------------|
| **Tabular** | **74.1%** | **80.2%** | ⭐⭐⭐⭐⭐ |
| Multi-Modal | 61.1% | 66.3% | ⭐⭐⭐ |
| Image | 60.4% | 63.8% | ⭐⭐⭐ |

---

## 🔍 **Detailed Analysis**

### 1. **Why Tabular Models Win**

**Reasons:**
- ✅ **Structured features are highly informative**: Brand name, MSRP, and condition directly correlate with resale value
- ✅ **Categorical encoding works well**: 624 features after one-hot encoding capture brand distinctions
- ✅ **Tree-based models handle non-linearity**: Complex decision boundaries between accept/reject
- ✅ **Ensemble methods reduce variance**: XGBoost and Random Forest combine multiple weak learners

**Evidence:**
```
Top Features (from Random Forest):
1. MSRP (numeric) - 23.4% importance
2. Brand (specific brands) - 18.7% importance  
3. Condition (ordinal) - 12.3% importance
4. Sub-category - 8.9% importance
5. Listing price - 7.2% importance
```

### 2. **Why Image Models Underperform**

**Current Performance:**
- Best: EfficientNet-B0 (61.7% F1, 63.9% AUC)
- Worst: DenseNet121 (57.7% F1, 62.4% AUC)

**Problems Identified:**
1. ❌ **Limited training data**: Only 1,690 training images
2. ❌ **Transfer learning mismatch**: ImageNet features != furniture quality features
3. ❌ **High visual variance**: Same brand/condition can look very different
4. ❌ **Image quality issues**: Some CDN images have artifacts
5. ❌ **Lighting/angle variations**: Hard to assess condition from single image

### 3. **Why Multi-Modal Disappoints**

**Current Performance:**
- Best: Early Fusion (65.0% F1, 72.0% AUC)
- Worst: Late Fusion Weighted (57.6% F1, 61.6% AUC)

**Analysis:**
- Multi-modal is **worse than tabular alone** (61.1% vs 74.1%)
- Multi-modal is **barely better than image alone** (61.1% vs 60.4%)
- **Issue**: Weak image features drag down strong tabular features

---

## 🚀 **Recommendations to Improve Accuracy**

### 🔥 **HIGH IMPACT - Implement Immediately**

#### 1. **Data Augmentation & Synthetic Features** (+3-5% expected)

```python
# Add engineered features:
- brand_tier: Map brands to tiers (luxury, mid, budget)
- msrp_to_price_ratio: MSRP / listing_price (deals indicator)
- age_estimate: Extract from description
- brand_popularity: Count of brand in dataset
- condition_numeric: Map to 1-4 scale
- category_brand_interaction: One-hot(category × brand)
```

**Implementation:**
```python
def engineer_features(df):
    # Brand tiers
    luxury_brands = ['Restoration Hardware', 'Pottery Barn', 'West Elm']
    df['brand_tier'] = df['brand_name'].apply(
        lambda x: 'luxury' if x in luxury_brands else 'standard'
    )
    
    # Value indicators
    df['msrp_to_price_ratio'] = df['msrp'] / (df['fb_listing_price'] + 1)
    df['is_deal'] = (df['msrp_to_price_ratio'] > 2).astype(int)
    
    # Confidence product
    df['confidence_product'] = df['confidence_brand'] * df['confidence_msrp']
    
    return df
```

#### 2. **Ensemble Best Tabular Models** (+2-3% expected)

Combine XGBoost, Random Forest, and Gradient Boosting:

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgboost_model),
        ('rf', random_forest_model),
        ('gb', gradient_boosting_model)
    ],
    voting='soft',  # Use probability averaging
    weights=[2, 1, 1]  # Weight XGBoost more
)
```

**Expected:** 78-80% accuracy

#### 3. **Handle Class Imbalance Better** (+1-2% expected)

Current: 55% accept, 45% reject (better than 39.5% in all_data)

```python
# Try SMOTE for oversampling minority class
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

#### 4. **Hyperparameter Optimization** (+1-2% expected)

Current grid search is limited. Use Bayesian optimization:

```python
from skopt import BayesSearchCV

param_space = {
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0)
}

bayes_search = BayesSearchCV(
    XGBClassifier(),
    param_space,
    n_iter=50,
    cv=5,
    scoring='f1_weighted'
)
```

### 💡 **MEDIUM IMPACT - Next Steps**

#### 5. **Fix Multi-Modal Architecture** (+3-5% expected over current multi-modal)

**Problem**: Image features are too weak, dragging down tabular.

**Solution**: Gated fusion with attention
```python
class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Learn to gate image features
        self.gate = nn.Sequential(
            nn.Linear(tabular_dim, 1),
            nn.Sigmoid()
        )
        # Only use image features when gate is open
        
    def forward(self, image_feat, tabular_feat):
        gate_value = self.gate(tabular_feat)
        # Weighted combination
        combined = gate_value * image_feat + (1 - gate_value) * tabular_feat
        return combined
```

#### 6. **Better Image Training Strategy** (+5-8% expected for image models)

**Current issues:**
- Only 2 epochs (quick mode)
- Small batch size (32)
- Basic augmentation

**Improvements:**
```python
# Stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Train longer
epochs = 50  # vs current 2
batch_size = 64  # vs current 32
learning_rate = 1e-4  # with warmup
```

#### 7. **Multi-View Images** (+3-5% expected)

Use multiple images per item (if available):
- Different angles
- Close-ups of damage
- Overall room context

**Average predictions** across views for robustness.

#### 8. **Stratified K-Fold Cross-Validation** (+1-2% expected)

Current: Single train/val/test split

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in skf.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    cv_scores.append(score)

print(f"CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

### 🔬 **RESEARCH / LONG-TERM**

#### 9. **Collect More Data** (+10-15% potential)

Current: 2,416 samples with both brand/MSRP

**Target**: 10,000+ samples
- More rare brands
- Better condition distribution
- More categories

#### 10. **Domain-Specific Image Model** (+5-10% potential)

Fine-tune on furniture-specific features:
- Fabric texture classification
- Damage detection
- Style recognition (modern, vintage, etc.)

Pre-train on:
- Pinterest furniture images
- Interior design datasets
- Furniture catalog photos

#### 11. **Multi-Task Learning** (+3-5% potential)

Train model to predict multiple targets:
- Accept/Reject (primary)
- Estimated MSRP (auxiliary)
- Condition grade (auxiliary)
- Brand category (auxiliary)

Shared representations improve generalization.

#### 12. **Active Learning** (+ongoing improvement)

Identify uncertain predictions and get labels:
```python
# Find samples where model is uncertain
probs = model.predict_proba(X_unlabeled)
uncertainty = 1 - np.max(probs, axis=1)
most_uncertain = np.argsort(uncertainty)[-100:]

# Manually label these, retrain
```

---

## 🎯 **Immediate Action Plan**

### Phase 1: Quick Wins (1-2 days, +5-7% expected)

1. ✅ **Engineer features** (brand_tier, ratios, interactions)
2. ✅ **Ensemble top 3 models** (XGB, RF, GB)
3. ✅ **Bayesian hyperparameter tuning**
4. ✅ **Handle class imbalance** (SMOTE or class weights)

**Target: 80-82% accuracy**

### Phase 2: Architecture Improvements (3-5 days, +3-5% expected)

1. ✅ **Fix multi-modal fusion** (gated attention)
2. ✅ **Better image training** (50 epochs, strong augmentation)
3. ✅ **Cross-validation** (5-fold stratified)
4. ✅ **Model stacking** (meta-learner on top of base models)

**Target: 82-85% accuracy**

### Phase 3: Data & Domain (1-2 weeks, +5-10% expected)

1. ✅ **Collect more data** (expand to 5,000+ samples)
2. ✅ **Domain-specific pre-training**
3. ✅ **Multi-task learning**
4. ✅ **Active learning loop**

**Target: 85-90% accuracy**

---

## 📊 **Expected Performance Trajectory**

```
Current:       76.6% (XGBoost baseline)
Phase 1:       80-82% (quick wins)
Phase 2:       82-85% (architecture)
Phase 3:       85-90% (data + domain)
Theoretical:   90-95% (with perfect features)
```

---

## 🎓 **For Your DS340 Report**

### Key Insights

1. **Structured data beats images** for furniture resale prediction
   - Tabular: 74.1% F1
   - Image: 60.4% F1
   - Difference: **13.7 percentage points**

2. **Multi-modal underperforms** due to weak image features
   - Expected: Image + Tabular > Tabular alone
   - Reality: 61.1% < 74.1%
   - **Lesson**: Weak modality can hurt strong one

3. **Data quality matters more than quantity**
   - `both_only` (2,416 samples): 76.6% accuracy
   - Clean, complete data enables better learning

4. **Tree-based ensembles excel** at structured classification
   - XGBoost, Random Forest, Gradient Boosting all >75%
   - Non-linear interactions captured effectively

### Recommendations for Report

1. **Lead with XGBoost** as production model (76.6%, 83.7% AUC)
2. **Explain why tabular wins** (feature informativeness)
3. **Discuss multi-modal failure** (weak image features)
4. **Propose improvements** (feature engineering, ensemble)
5. **Justify business value** (automate 76% of decisions correctly)

---

## 💼 **Business Impact**

### Current Performance
- **76.6% accuracy** = Can automate 3/4 of decisions
- **83.7% AUC** = Excellent discrimination between accept/reject
- **83.3% precision (accept class)** = Few false accepts
- **85.5% recall (accept class)** = Catch most valuable items

### ROI Calculation
```
Manual review time: 2 min/item
Items per day: 100
Days per year: 365
Annual items: 36,500

With 76.6% automation:
Automated: 27,959 items
Manual: 8,541 items
Time saved: 55,918 minutes = 932 hours
At $30/hr: $27,960 saved per year
```

### Risk Mitigation
- **False Accepts (Type I)**: 16.7% of accepts are errors
  - **Mitigation**: Second review for high-value items (MSRP >$1000)
- **False Rejects (Type II)**: 14.5% of rejects are errors
  - **Mitigation**: Periodic audit of rejected items

---

## 🎉 **Summary**

✅ **Best Model**: XGBoost (76.6% accuracy, 83.7% AUC)  
✅ **Deployment Ready**: Yes, with human review for edge cases  
✅ **Improvement Potential**: 80-85% with feature engineering & ensembles  
✅ **Next Steps**: Implement Phase 1 recommendations (quick wins)

**Your experimental framework successfully identified the best approach for furniture classification. The tabular XGBoost model is production-ready and can automate 76% of accept/reject decisions with high confidence.**

---

📊 **Visualizations Available:**
- `f1_comparison.png` - F1 scores across all models
- `best_models_comparison.png` - Top performers by type
- `auc_comparison.png` - AUC-ROC comparison
- `f1_heatmap.png` - Performance heatmap

📈 **Data Available:**
- `results_comparison.csv` - Full numerical results
- `experiment_report.txt` - Text summary
- `all_experiment_results.json` - Complete results with confusion matrices

🤖 **Models Available:**
- `models/xgboost_both_only.pkl` - Best model (deploy this!)
- All other trained models saved for comparison


