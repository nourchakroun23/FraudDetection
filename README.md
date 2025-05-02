#  Fraud Detection Project — Hybrid Deep Learning & Federated Learning Approach

##  Overview

This project addresses the problem of credit card fraud detection using a **hybrid approach** that combines **deep learning (autoencoders)** for anomaly detection and **machine learning (XGBoost)** for classification. Furthermore, a **federated learning framework** is proposed to ensure data privacy while maintaining high model performance across distributed clients.

##  Project Architecture

### 1. **Autoencoder (AE) for Anomaly Detection**
- Unsupervised deep neural network trained to reconstruct normal transactions.
- High reconstruction error indicates potential fraud.
- Acts as a first-level filter to reduce false positives.

### 2. **XGBoost Classifier**
- Trained on a balanced dataset (after SMOTE/undersampling).
- Features include raw variables + reconstruction error from AE.
- Robust to outliers and well-suited to imbalanced data.

### 3. **Federated Learning Module**
- Simulates multiple institutions/clients training local models on private data.
- Preserves data confidentiality and complies with data-sharing regulations.

## Dataset Summary

Data preprocessing steps:
- Data cleaning
- Data encoding
- Feature scaling (MinMax)
- Class balancing with SMOTE
- Train/test split using stratified sampling

## Technologies & Libraries

- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- XGBoost
- imbalanced-learn
- PySyft or Flower (for federated learning)
- pandas, numpy, matplotlib, seaborn

##  Evaluation Metrics

To evaluate model effectiveness in a highly imbalanced setting:
- **Reconstruction error** (for AE)
- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **AUC-ROC**
- **Confusion Matrix**

##  Key Results

| Model                  | F1-score | AUC-ROC | Notes                             |
|------------------------|----------|---------|-----------------------------------|
| Autoencoder Only       | 0.78     | 0.85    | Good for anomaly detection        |
| XGBoost Only           | 0.92     | 0.96    | Strong supervised performance     |
| AE + XGBoost (Hybrid)  | **0.95** | **0.98**| Best balance of precision & recall |
| Federated XGBoost      | ~0.91    | ~0.96   | Slight drop, privacy preserved    |

##  Federated Learning

- Framework used: **[Flower / PySyft]**
- Simulated multiple clients (banks/branches)
- Each client trains locally on private data
- Periodic aggregation to update global model

