# 💳 Fraud Detection Project — Hybrid Deep Learning & Federated Learning Approach

## 📘 Overview

This project addresses the problem of credit card fraud detection using a **hybrid approach** that combines **deep learning (autoencoders)** for anomaly detection and **machine learning (XGBoost)** for classification. Furthermore, a **federated learning framework** is proposed to ensure data privacy while maintaining high model performance across distributed clients.

## 🧠 Project Architecture

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
- Uses **Federated Averaging (FedAvg)** to update a global model.
- Preserves data confidentiality and complies with data-sharing regulations.

## 📊 Dataset Summary

The main dataset is a real-world credit card transaction log:
- 284,807 transactions
- 492 fraudulent cases (~0.17%)
- Features: anonymized numerical inputs (V1–V28), `Time`, `Amount`, `Class`

Data preprocessing steps:
- Feature scaling (MinMax or RobustScaler)
- Class balancing with SMOTE
- Outlier filtering (optional)
- Train/test split using stratified sampling

## 📦 Project Structure

