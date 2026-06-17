# Fraud Detection & Analysis using Python, XGBoost and Power BI

## Project Overview

This project analyzes over 1 million financial transactions to identify fraud patterns, engineer risk indicators, and build a machine learning model capable of predicting fraudulent transactions.

The objective was not only to build a fraud detection model, but also to generate actionable business insights and visualize them through an interactive Power BI dashboard.

---

## Business Problem

Financial institutions process millions of transactions every day, making it difficult to manually identify fraudulent activity.

Fraudulent transactions often exhibit patterns such as:

* Unusual transaction timing
* High transaction velocity
* High-risk merchant categories
* Large transaction amounts
* Suspicious transaction locations

The goal of this project was to identify these patterns and create a system capable of flagging potentially fraudulent transactions.

---

## Dataset

Dataset: Fraud Detection Dataset (Kaggle)

* Total Transactions: ~1,048,575
* Fraud Transactions: ~6,000
* Fraud Rate: ~0.58%

Key Fields:

* Transaction Amount
* Merchant Category
* Transaction Timestamp
* Customer Location
* Merchant Location
* Fraud Label

---

## Exploratory Data Analysis

The following analyses were performed:

### Category-wise Fraud Analysis

Identified merchant categories with the highest fraud rates.

Key Findings:

* Shopping transactions exhibited elevated fraud risk.
* Miscellaneous online transactions showed higher fraud occurrence.
* Certain merchant categories experienced significantly higher fraud rates than the dataset average.

### Time-based Analysis

Analyzed fraud occurrence across different hours of the day.

Key Findings:

* Fraud activity peaks during late-night hours.
* Significant fraud spikes observed between 12 AM – 3 AM.
* Secondary increase observed during late evening hours.

### Distance Analysis

Calculated geographical distance between customer and merchant locations using the Haversine Formula.

Key Findings:

* Average transaction distance alone was not a strong predictor of fraud.
* Distance became more useful when combined with other behavioral indicators.

### Velocity Analysis

Created transaction velocity features:

* Transactions in Last 1 Hour
* Transactions in Last 24 Hours
* Time Difference Between Consecutive Fraud Transactions

Key Findings:

* Fraudsters frequently perform multiple transactions within short time intervals.
* Velocity-based indicators significantly improved fraud detection capability.

---

## Feature Engineering

Custom features created:

* High Risk Merchant Category Flag
* High Value Transaction Flag
* Night Transaction Flag
* Transaction Velocity Features
* Distance-Based Features
* Fraud Probability Score

---

## Machine Learning Model

Model Used:

* XGBoost Classifier

Class imbalance was handled using:

* scale_pos_weight

Evaluation Metrics:

* Classification Report
* ROC-AUC Score

### Results

| Metric           | Value |
| ---------------- | ----- |
| AUC-ROC          | 0.984 |
| Fraud Recall     | 0.88  |
| Overall Accuracy | 0.97  |

The model achieved strong fraud detection performance while maintaining high discrimination capability between fraudulent and legitimate transactions.

---

## Power BI Dashboard

The Power BI dashboard was created to provide business-friendly insights.

### Executive Summary

* Total Transactions
* Total Fraud Transactions
* Fraud Rate
* Total Fraud Amount

### Fraud Pattern Analysis

* Fraud by Category
* Fraud by Hour
* Fraud Loss by Merchant Category
* Risk Distribution

### Model Insights

* Fraud Probability Distribution
* Velocity-Based Risk Analysis
* High-Risk Transaction Monitoring

---

## Key Business Insights

1. Fraud activity is concentrated during late-night hours.
2. Shopping and online merchant categories exhibit higher fraud risk.
3. Transaction velocity is a strong indicator of fraudulent activity.
4. Fraudsters often perform multiple transactions within short time windows.
5. Combining behavioral features significantly improves fraud detection capability.

---

## Technologies Used

### Data Analysis

* Python
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Power BI

### Machine Learning

* XGBoost
* SHAP


## Future Improvements

* Real-time fraud monitoring
* Explainable AI dashboard using SHAP
* Streamlit deployment
* Automated fraud alerting system
* Advanced anomaly detection techniques

---

## Author

CodeOverlord11
