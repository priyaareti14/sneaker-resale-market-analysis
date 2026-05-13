# Sneaker Resale Market Analysis

## Overview
This project analyzes 99,956 StockX transactions across Yeezy and Off-White sneakers to identify the key factors that drive resale prices above retail on the secondary market.

## Business Question
What factors drive sneaker resale prices above retail on the secondary market?

## Tools and Libraries
- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (Logistic Regression, Linear Regression, Decision Tree)
- imbalanced-learn (SMOTE)

## Project Workflow
1. Data cleaning and type conversion
2. Feature engineering (Days Since Release, Markup %, Price Category, Log Sale Price)
3. Exploratory Data Analysis
4. Predictive modeling with three approaches
5. Model evaluation and interpretation

## Models and Results

| Model | Metric | Score |
|---|---|---|
| Logistic Regression | ROC-AUC | 0.78 |
| Linear Regression | R-squared | 0.44 |
| Decision Tree | Accuracy | 90.6% |

## Key Findings
- Days Since Release was the strongest predictor of resale premium, with feature importance of 0.52
- Off-White brand added approximately $378 to predicted sale price compared to Yeezy
- SMOTE was applied to handle class imbalance in the training data

## Dataset
StockX Data Contest 2019, publicly available on Kaggle:
https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest
