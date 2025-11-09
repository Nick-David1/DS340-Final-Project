# 🛋️ DS340: Sofa Saleability Prediction

**Team:** Nick David & Adrian Dybacki  
**Client:** second-edition.co  
**Course:** DS340 Final Project

---

## 📋 Project Overview

This repository contains a **multi-modal machine learning model** built with PyTorch to predict the saleability of consignment furniture. Our model combines:
- 🖼️ **Image Data** (CNN-based feature extraction)
- 📊 **Tabular Data** (structured metadata like brand, condition, MSRP)

**Goal:** Automate the classification of furniture submissions as "Accept" or "Reject" to streamline the consignment intake process.

---

## ⚠️ CRITICAL: DATA PRIVACY

This project uses **proprietary data** provided by our client.

🚨 **DO NOT COMMIT RAW DATA OR IMAGES** 🚨

- All sensitive data is stored locally in the `/data` directory (blocked by `.gitignore`)
- The `.gitignore` is configured to protect:
  - `data/sofa_data.csv` (accepted items)
  - `data/rejected_data.csv` (rejected items)
  - `data/images/` (all product photos)
  - `models/*.pth` (trained model checkpoints)

✅ Only `data/sample_data.csv` (a small, anonymized sample) may be committed for demonstration purposes.

---

## 🗂️ Project Structure

```
sofa-consignment-ai/
│
├── .gitignore              # Protects sensitive data
├── README.md               # This file
├── requirements.txt        # Python dependencies
│
├── data/                   # ⚠️ NOT COMMITTED TO GIT
│   ├── sofa_data.csv       # Full accepted dataset (private)
│   ├── rejected_data.csv   # Full rejected dataset (private)
│   ├── sample_data.csv     # Small safe sample (can commit)
│   └── images/             # Downloaded product images (private)
│
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_EDA_and_Data_Cleaning.ipynb
│   └── 02_Model_Training_and_Evaluation.ipynb
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── download_images.py  # Script to download images from URLs
│   ├── dataset.py          # Custom PyTorch Dataset class
│   ├── model.py            # Multi-modal neural network architecture
│   ├── train.py            # Training and evaluation script
│   └── utils.py            # Helper functions
│
└── models/                 # ⚠️ NOT COMMITTED TO GIT
    └── *.pth               # Saved model checkpoints
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/sofa-consignment-ai.git
cd sofa-consignment-ai
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add Your Data

Place the client-provided data files in the `/data` directory:
- `sofa_data.csv`
- `rejected_data.csv`

### 5. Download Images

Run the image downloader script:

```bash
python src/download_images.py
```

---

## 📊 Workflow

### Step 1: Exploratory Data Analysis (EDA)

Open and run the EDA notebook:

```bash
jupyter notebook notebooks/01_EDA_and_Data_Cleaning.ipynb
```

### Step 2: Model Training

Train the multi-modal model:

```bash
python src/train.py --epochs 20 --batch_size 32 --learning_rate 0.001
```

### Step 3: Evaluation

View results and metrics in:

```bash
jupyter notebook notebooks/02_Model_Training_and_Evaluation.ipynb
```

---

## 🧪 Experiments

### Experiment 1: Value of Multi-modality
Compare three model architectures:
- **Model A:** Image-only (CNN)
- **Model B:** Tabular-only (MLP)
- **Model C:** Fused (CNN + MLP) ⭐

**Hypothesis:** The fused model will significantly outperform single-modality baselines.

### Experiment 2: Image Augmentation
Test augmentation strategies:
- **Setting 1:** No augmentation
- **Setting 2:** Light augmentation (flips, brightness)
- **Setting 3:** Heavy augmentation (flips, rotation, color jitter) ⭐

**Hypothesis:** Heavy augmentation will improve generalization.

### Experiment 3: Backbone Architecture
Compare pre-trained CNNs:
- **Setting 1:** ResNet50 (baseline)
- **Setting 2:** MobileNetV3 (efficiency)
- **Setting 3:** EfficientNetB0 (accuracy) ⭐

**Hypothesis:** EfficientNetB0 will achieve the highest F1-score.

---

## 📈 Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** Avoiding false accepts
- **Recall:** Not missing valuable items
- **F1-Score:** Balanced measure (primary metric)

**Why F1?** In this business context, both false positives (wasted pickup trips) and false negatives (missed profitable items) are costly.

---

## 👥 Team & Collaboration

### Adding Collaborators

Repository owner should:
1. Go to **Settings** → **Collaborators**
2. Add your partner's GitHub username
3. They'll receive an invitation email

### Best Practices
- ✅ Create feature branches for major changes
- ✅ Use descriptive commit messages
- ✅ Pull before you push
- ✅ Never commit sensitive data
- ✅ Test code before pushing

---

## 📅 Project Timeline

- **Weeks 1-2 (by Nov 4):** Data acquisition & EDA
- **Weeks 3-4 (by Nov 18):** Milestone 1 - Baseline image model ✅
- **Weeks 5-6 (by Dec 2):** Multi-modal integration & tuning
- **Week 7 (by Dec 8):** Final experiments & presentation prep

---

## 🛠️ Tech Stack

- **Framework:** PyTorch
- **Data Processing:** pandas, NumPy
- **Preprocessing:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebooks:** Jupyter

---

## 📝 License

This is a private academic project. All rights reserved to the project team and client.

---

## 🤝 Acknowledgments

Special thanks to **second-edition.co** for providing the dataset and problem context.

---

## 📧 Contact

For questions or issues, contact:
- Nick David
- Adrian Dybacki

