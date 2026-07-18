<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=250&section=header&text=IEEE-CIS%20Fraud%20Detection&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Transaction%20Risk%20Analysis%20System&descAlignY=55&descAlign=50" width="100%"/>

<!-- Animated Typing Badge -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=667EEA&center=true&vCenter=true&width=600&lines=Detecting+Financial+Fraud+with+Machine+Learning;LightGBM+%7C+XGBoost+%7C+CatBoost;From+434+Features+to+Intelligent+Risk+Scoring;Kaggle+IEEE-CIS+Competition+Pipeline" alt="Typing SVG" />
</a>

<!-- Status Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge&logo=statuspage&logoColor=white&labelColor=1a1a2e" />
  <img src="https://img.shields.io/badge/F1%20Score-0.767-blue?style=for-the-badge&logo=lightgbm&logoColor=white&labelColor=1a1a2e" />
  <img src="https://img.shields.io/badge/Features-442-orange?style=for-the-badge&logo=databricks&logoColor=white&labelColor=1a1a2e" />
  <img src="https://img.shields.io/badge/Dataset-590K%2B%20Transactions-ff6b6b?style=for-the-badge&logo=database&logoColor=white&labelColor=1a1a2e" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" />
  <img src="https://img.shields.io/badge/License-MIT-9cf?style=for-the-badge&logo=opensourceinitiative&logoColor=white&labelColor=1a1a2e" />
</p>

<!-- Animated Divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

---

## 🎯 Project Overview

> **An end-to-end fraud detection pipeline built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition dataset.**

This repository demonstrates a complete machine learning workflow for detecting fraudulent transactions in real-time financial systems. Starting with **590K+ transactions** and **434 raw features**, we engineer intelligent risk signals, benchmark multiple gradient boosting algorithms, and maintain a rigorous experiment log — all designed for production-ready deployment.

### 🚀 What Makes This Different?

| Aspect | Traditional Approach | This Pipeline |
|--------|---------------------|---------------|
| Feature Engineering | Guess & pray | Hypothesis-driven with adversarial validation |
| Model Selection | Single algorithm | LGBM · XGBoost · CatBoost benchmark |
| Validation | Naïve train/test | Stratified splits + early stopping + threshold tuning |
| Experiment Tracking | Scattered notebooks | Versioned experiment log with F1 deltas |
| Categorical Handling | One-hot explosion (485+ features) | Native LightGBM category support |

---

## 📊 Live Experiment Dashboard

<!-- Animated Stats Grid -->
<div align="center">

| 🏆 **Best F1** | ⚡ **Best Threshold** | 🧠 **Model** | 🔢 **Features** | 📈 **Improvement** |
|:---:|:---:|:---:|:---:|:---:|
| **0.767** | **0.25** | **LightGBM** | **442** | **+27.4%** over baseline |

</div>

<details>
<summary>🔬 <b>Click to Expand Full Experiment History</b></summary>

| Version | Experiment | F1 Score | Δ F1 | Status |
|:-------:|------------|:--------:|:----:|:------:|
| V1 | Random Forest Baseline | 0.602 | — | 🟡 Baseline |
| V2 | RF + Balanced Weights | 0.570 | -0.032 | 🔴 Rejected |
| V3 | **LightGBM Baseline** | ~0.740 | +0.14 | 🟢 Adopted |
| V4 | Threshold Tuning | **0.767** | +0.02 | 🟢 Keep |
| V5 | Time Features (`trans_day`, `trans_weekday`) | 0.767 | ~0.000 | 🔴 Removed |
| V6 | UID Statistics | 0.767+ | Positive | 🟢 Keep |
| V7 | Velocity Features | 0.743 | -0.024 | 🔴 Removed |
| V8 | UID2 (`card1 + addr1 + D1`) | 0.696 | -0.071 | 🔴 Rejected |
| V9 | Native Categorical + Cleanup | **0.767** | — | 🟢 **Current** |

</details>

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    A[📁 Raw Dataset<br/>590K+ transactions] --> B[🔧 Feature Engineering]
    B --> C[⚡ UID-Based Features<br/>card1 + addr1 aggregations]
    B --> D[🧹 Cleanup<br/>Remove harmful features]
    C --> E[🤖 Model Training]
    D --> E
    E --> F[🌲 LightGBM]
    E --> G[⚡ XGBoost]
    E --> H[🐱 CatBoost]
    F --> I[📊 Threshold Tuning]
    G --> I
    H --> I
    I --> J[🏆 Best Model Selection]
    J --> K[📈 Feature Importance Analysis]
    K --> L[🔮 Fraud Prediction API]

    style A fill:#1a1a2e,stroke:#667eea,stroke-width:2px,color:#fff
    style E fill:#16213e,stroke:#667eea,stroke-width:2px,color:#fff
    style J fill:#0f3460,stroke:#e94560,stroke-width:3px,color:#fff
    style L fill:#e94560,stroke:#fff,stroke-width:2px,color:#fff
```

---

## 📁 Dataset Structure

<!-- Animated Dataset Card -->
<div align="center">

| File | Records | Columns | Purpose |
|------|:-------:|:-------:|---------|
| `train_transaction.csv` | 590K+ | 394 | Main transaction data + target |
| `train_identity.csv` | ~144K | 41 | Device & identity metadata |
| **Merged Total** | 590K+ | **~434** | Complete feature set |

</div>

### 🔍 Feature Categories

<details>
<summary><b>🎯 Target Variable (1)</b></summary>

- **`isFraud`** — Binary target. `1` = fraudulent transaction

</details>

<details>
<summary><b>💳 Transaction Identity (5)</b></summary>

| Feature | Description |
|---------|-------------|
| `TransactionID` | Unique transaction identifier |
| `TransactionDT` | Time delta (seconds) from reference |
| `TransactionAmt` | Dollar amount — key risk signal |
| `ProductCD` | Product category (W, H, R, S, C) |

</details>

<details>
<summary><b>💳 Card Features (6) — <code>card1-card6</code></b></summary>

Anonymized payment card metadata. Fraudsters often test multiple cards; repeated card usage indicates trust.

</details>

<details>
<summary><b>📍 Address & Distance (4) — <code>addr1/2</code>, <code>dist1/2</code></b></summary>

Billing address and geographic distance features. Address mismatch and long-distance transactions are strong fraud signals.

</details>

<details>
<summary><b>📧 Email Domains (2) — <code>P_emaildomain</code>, <code>R_emaildomain</code></b></summary>

Payer and recipient email domains. Free vs. corporate domains carry different risk weights.

</details>

<details>
<summary><b>🔢 Count Features (14) — <code>C1-C14</code></b></summary>

Behavioral frequency counts: card usage frequency, address transaction counts, unique device patterns. Low counts = new/unusual = higher risk.

</details>

<details>
<summary><b>⏱️ Time Delta Features (15) — <code>D1-D15</code></b></summary>

Time differences capturing: time since first card use, time since last transaction, account dormancy periods. Small deltas = fresh accounts = risky.

</details>

<details>
<summary><b>✅ Match Features (9) — <code>M1-M9</code></b></summary>

Boolean matching indicators: card address matches billing, device matches history, shipping address validation. Mismatches = fraud signal.

</details>

<details>
<summary><b>🔐 Identity/Device Features (40) — <code>id_01-id_38</code>, <code>DeviceType</code>, <code>DeviceInfo</code></b></summary>

Device fingerprinting, OS/browser signatures, behavioral biometrics. Device switching patterns are critical fraud indicators.

</details>

<details>
<summary><b>🧬 Anonymized Features (339) — <code>V1-V339</code></b></summary>

Pre-engineered features likely from PCA, polynomial interactions, and risk scoring. Direct interpretation is abstract; feature importance analysis required.

</details>

---

## 🛠️ Engineered Features

<!-- Feature Cards -->
<div align="center">

| Feature | Type | Purpose | Status |
|---------|------|---------|:------:|
| `uid` | 🆔 Identifier | `card1 + addr1` composite key | 🟢 Active |
| `uid_transaction_count` | 📊 Count | Previous transactions per UID | 🟢 Active |
| `uid_mean_amount` | 💰 Statistic | Historical mean transaction amount | 🟢 Active |
| `uid_std_amount` | 📈 Statistic | Historical std dev of amounts | 🟢 Active |
| `uid_amount_ratio` | ⚖️ Ratio | Current amount / historical mean | 🟢 Active |
| `uid_amount_zscore` | 🎯 Score | Standardized amount anomaly | 🟢 Active |
| `uid_mean_hour` | 🕐 Temporal | Average transaction hour per UID | 🟢 Active |
| `trans_day` | 📅 Temporal | Day of transaction | 🔴 Removed |
| `trans_weekday` | 📅 Temporal | Weekday index | 🔴 Removed |
| `velocity_1h` | ⚡ Velocity | Transactions in last hour | 🔴 Removed |
| `velocity_24h` | ⚡ Velocity | Transactions in last 24h | 🔴 Removed |
| `uid2_*` | 🧪 Experimental | `card1 + addr1 + D1` composite | 🔴 Removed |

</div>

> **💡 Why we removed features:** Every feature above marked 🔴 went through **ablation testing**. If it didn't improve F1 or actively hurt it (like `uid2_*` which crashed F1 to 0.696), it was deleted. Winners don't hoard features — they validate hypotheses.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
# pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ieee-fraud-detection.git
cd ieee-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place train_transaction.csv and train_identity.csv in data/raw/
```

### Running the Pipeline

```bash
# 1. Data preprocessing & optimization
python src/preprocess.py

# 2. Feature engineering
python src/feature_engineering.py

# 3. Model training & benchmarking
python src/train.py --models lgbm xgboost catboost

# 4. Threshold tuning & evaluation
python src/evaluate.py --threshold-search
```

### One-Line Prediction

```python
import joblib

model = joblib.load('models/best_lgbm_model.pkl')
# Returns fraud probability (0-1)
probability = model.predict_proba(your_transaction_df)[:, 1]
```

---

## 📈 Model Performance

<!-- Performance Comparison Table -->
<div align="center">

### Threshold Evaluation (LightGBM)

| Threshold | Recall | Precision | F1 Score | Use Case |
|:---------:|:------:|:---------:|:--------:|----------|
| 0.10 | 0.796 | 0.599 | 0.683 | High recall (catch almost all fraud) |
| 0.15 | 0.758 | 0.733 | 0.745 | Balanced detection |
| 0.20 | 0.726 | 0.809 | 0.765 | Strong precision |
| **0.25** ⭐ | **0.693** | **0.859** | **0.767** | **🏆 Optimal F1** |
| 0.30 | 0.669 | 0.895 | 0.766 | High precision |
| 0.40 | 0.615 | 0.932 | 0.741 | Conservative flagging |
| 0.50 | 0.569 | 0.954 | 0.712 | Minimal false positives |

</div>

### Algorithm Benchmark (Coming Soon)

| Model | F1 Score | Training Time | Inference Speed | Best For |
|-------|:--------:|:-------------:|:---------------:|----------|
| LightGBM | **0.767** | ⚡ Fast | ⚡ Fast | Current production choice |
| XGBoost | TBD | TBD | TBD | Under evaluation |
| CatBoost | TBD | TBD | TBD | Under evaluation |

> **🔄 Next Update:** Full 3-model benchmark with frozen features for fair comparison.

---

## 🧠 Key Insights

### What Makes a Transaction Risky? 🚨

```
1. NEW/RARE COMBINATIONS  → First time seeing card + address + device
2. MISMATCH SIGNALS       → Address mismatch, device mismatch, domain mismatch  
3. SUDDEN CHANGES         → Device changes, geographic jumps, spending spikes
4. LOW FREQUENCY          → New user, new card, new address (no history)
5. IMPOSSIBLE PATTERNS    → iPhone + Chrome + Windows (device inconsistency)
6. AMOUNT ANOMALY         → Transaction >> historical average
7. TEMPORAL ANOMALY       → Transaction at unusual hour
```

### What Makes a Transaction Trustworthy? ✅

```
1. HIGH FREQUENCY         → Repeated card, address, device usage
2. CONSISTENCY            → Everything matches historical pattern
3. CONTINUITY             → Small time gaps (regular user behavior)
4. DEVICE STABILITY       → Same device used repeatedly
5. PATTERN MATCH          → All M1-M9 features align
```

---

## 🗂️ Repository Structure

```
ieee-fraud-detection/
├── 📁 data/
│   ├── raw/              # Original Kaggle CSVs
│   ├── processed/        # Optimized pickle files
│   └── external/         # Supplementary data
├── 📁 notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_baseline.ipynb # Random Forest baseline
│   └── 03_lightgbm.ipynb # Current best pipeline
├── 📁 src/
│   ├── preprocess.py     # Data cleaning & optimization
│   ├── features.py       # Feature engineering pipeline
│   ├── train.py          # Model training & benchmarking
│   └── evaluate.py       # Threshold tuning & metrics
├── 📁 models/
│   ├── best_lgbm.pkl     # Production model
│   └── experiments/      # Versioned experiment artifacts
├── 📁 reports/
│   ├── figures/          # EDA plots & feature importance
│   └── experiment_log.md # Detailed experiment history
├── README.md             # You are here! 🎯
└── requirements.txt      # Python dependencies
```

---

## 🤝 How to Contribute

We welcome contributions! Here's how to get involved:

### 🐛 Found a Bug?

1. **Check** if the issue already exists in [Issues](https://github.com/yourusername/ieee-fraud-detection/issues)
2. **Open a new issue** with:
   - Clear description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (Python version, OS)

### 💡 Have an Idea?

- **Feature requests:** Open an issue with the `enhancement` label
- **New algorithms:** We are actively benchmarking XGBoost and CatBoost
- **Feature engineering:** Follow the hypothesis → evidence → feature → ablation workflow

### 🔧 Contribution Workflow

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/amazing-feature

# 3. Commit your changes
git commit -m 'Add amazing feature'

# 4. Push to branch
git push origin feature/amazing-feature

# 5. Open a Pull Request
```

### 📋 Contribution Guidelines

- **Code Style:** Follow PEP 8. We use `black` and `flake8`.
- **Experiments:** Every new feature must include ablation results in `reports/experiment_log.md`
- **Documentation:** Update README if you change the pipeline structure
- **Tests:** Add tests for new utility functions in `tests/`

---

## 📚 Citation & Acknowledgments

If you use this code or dataset analysis in your research, please cite:

```bibtex
@misc{ieee-fraud-detection-2026,
  title={IEEE-CIS Fraud Detection: End-to-End ML Pipeline},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/ieee-fraud-detection}},
  note={Kaggle IEEE-CIS Fraud Detection Competition}
}
```

**Dataset Source:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)  
**Competition Host:** IEEE Computational Intelligence Society  
**Original Data:** Vesta Corporation

---

## 📬 Contact & Support

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail)](mailto:your.email@example.com)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com/yourprofile)

</div>

---

<div align="center">

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=150&section=footer&text=Happy%20Fraud%20Hunting!&fontSize=30&fontColor=ffffff&animation=fadeIn" width="100%"/>

**⭐ Star this repo if it helped you!**  
*Built with 💜, LightGBM, and a lot of coffee.*

</div>
