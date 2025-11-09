# DS340: Sofa Saleability Prediction

Team: Nick David & Adrian Dybacki  
Client: second-edition.co  
Course: DS340 Final Project

## Project Overview

This repository contains a multi-modal machine learning model built with PyTorch to predict the saleability of consignment furniture. The model combines image data (CNN-based feature extraction) with tabular metadata (brand, condition, MSRP, dimensions, style) to classify furniture submissions as "Accept" or "Reject" for consignment pickup.

## Data Privacy Notice

This project uses proprietary data provided by our client. All sensitive data is stored locally in the /data directory and is blocked by .gitignore. Do not commit raw data files or images to version control.

Protected files:
- data/sofa_data.csv (accepted items)
- data/rejected_data.csv (rejected items)
- data/images/ (product photos)
- models/*.pth (trained model checkpoints)

Only data/sample_data.csv may be committed for demonstration purposes.

## Project Structure

```
.
├── data/
│   ├── sample_data.csv
│   └── images/
├── notebooks/
├── src/
├── models/
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Place data files in the /data directory

## Experiments

### Experiment 1: Value of Multi-modality
Compare the performance of three model architectures:
- Model A: Image-only (CNN)
- Model B: Tabular-only (MLP)
- Model C: Multi-modal (CNN + MLP fusion)

Hypothesis: The fused multi-modal model will outperform single-modality baselines.

### Experiment 2: Image Augmentation
Test three augmentation strategies:
- No augmentation
- Light augmentation (flips, brightness adjustments)
- Heavy augmentation (flips, rotations, color jitter, contrast shifts)

Hypothesis: Heavy augmentation will improve model generalization.

### Experiment 3: Backbone Architecture
Compare pre-trained CNN backbones:
- ResNet50 (baseline)
- MobileNetV3 (lightweight)
- EfficientNetB0 (modern architecture)

Hypothesis: EfficientNetB0 will achieve the highest F1-score while MobileNetV3 offers the best speed-accuracy tradeoff.

## Evaluation Metrics

Primary metric: F1-Score  
Secondary metrics: Accuracy, Precision, Recall

We prioritize F1-score because both false positives (wasted pickup trips) and false negatives (missed profitable items) are costly in this business context.

## Project Timeline

- Weeks 1-2: Data acquisition and exploratory data analysis
- Weeks 3-4: Baseline image model development (Milestone 1)
- Weeks 5-6: Multi-modal model integration and tuning
- Week 7: Final experiments and presentation preparation

## Technology Stack

- PyTorch (deep learning framework)
- pandas and NumPy (data manipulation)
- scikit-learn (preprocessing and evaluation)
- Matplotlib and Seaborn (visualization)
- Jupyter (interactive development)

