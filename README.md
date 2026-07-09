# SMS Spam Filtering — Machine Learning Classification

A machine learning system that detects spam SMS messages using classical ML models and a deep learning neural network, built for CSC-237 (Data Analytics) at Beaconhouse National University.

## Problem

Spam text messages are a daily nuisance — often carrying phishing links and scams. Manually blocking numbers doesn't work because spammers keep changing them. This project builds an automated classifier to flag spam messages, while carefully handling a **heavily imbalanced dataset** (4,825 real "ham" messages vs. only 747 spam messages) without letting synthetic oversampled data leak into the test set — a mistake that quietly inflates accuracy if you're not careful.

## Dataset

- **Source:** [UCI SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 labeled SMS messages (ham / spam)

## Approach

1. **Text cleaning** — lowercase, strip numbers/special characters, remove stopwords
2. **Vectorization** — TF-IDF (top 3,000 features)
3. **Class balancing** — SMOTE applied **only to the training split**, after the train/test split, to avoid data leakage
4. **Models compared:** Naive Bayes, Decision Tree, SVM, and a Multi-Layer Perceptron (MLP) neural network
5. **Robustness testing** — every model tested across 3 train/test splits (80-20, 70-30, 60-40) × 3 random seeds × 3 parameter values = **108 experiments total**

## Results

| Model | Best ROC-AUC | Notes |
|---|---|---|
| SVM | 0.9851 | Best raw separation, slowest to train |
| MLP (Neural Net) | 0.9846 | Best overall balance of accuracy + recall, most stable convergence |
| Naive Bayes | 0.9809 | Fast, reliable baseline |
| Decision Tree | 0.8881 | Weakest — struggled with 3,000 sparse TF-IDF features |

**Key finding:** the MLP neural network gave the best overall real-world performance — high accuracy *and* high recall on spam (meaning it actually catches spam messages, not just avoids false alarms on real ones).

## Tech Stack

Python · scikit-learn · TensorFlow/Keras · imbalanced-learn (SMOTE) · pandas · matplotlib

## Files

- `SMS_Spam_Filtering_Model.ipynb` — full code: preprocessing, training, evaluation, all 108 experiments
- `Report.pdf` — full written report with methodology, results, and discussion

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook SMS_Spam_Filtering_Model.ipynb
```

## Author

Momin Ahmad — BSc Computer Science, Beaconhouse National University
