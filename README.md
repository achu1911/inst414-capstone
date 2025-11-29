# INST414 Capstone Project: Customer Churn Prediction

**Author:** Alexa Chu  
**Course:** INST414 – Applied Data Science Pipeline (ADSP)  
**Date:** September 28, 2025  

## Overview
This project aims to predict customer churn using the IBM Telco Customer Churn dataset. The goal is to identify which customers are likely to leave and uncover the key factors influencing churn. Insights from this analysis can help businesses implement targeted retention strategies, improve customer satisfaction, and reduce revenue loss.

## Dataset
- **Primary:** IBM Telco Customer Churn ([Kaggle link](https://www.kaggle.com/blastchar/telco-customer-churn)), 7,043 records, 21 attributes  
- **Backup:** Bank Customer Churn ([Kaggle link](https://www.kaggle.com)), 10,000 records, 12 attributes  

## Approach
1. Exploratory Data Analysis (EDA) to identify patterns  
2. Feature selection to determine key drivers of churn  
3. Predictive modeling to estimate churn probability  
4. Evaluation using predictive accuracy and actionable insights  
5. Documentation of limitations and contingency plans  

## Project Goals
- Predict customer churn with reasonable accuracy  
- Identify the most influential factors driving churn  
- Provide actionable business recommendations  

## Sprint 2 Progress

- Acquired and cleaned the IBM Telco Customer Churn dataset (7,032 rows x 21 columns after cleaning)
- Handled missing values and standardized categorical variables
- Conducted exploratory data analysis (EDA) and identified key predictors of churn
- Refined problem statement to focus on short-tenure, month-to-month contract customers
- Prepared plan for predictive modeling, starting with logistic regression

## Sprint 3 Progress

- Implemented a full preprocessing pipeline with stratified train/validation/test split (70/15/15), one-hot encoding, and StandardScaler applied only to training data
- Established a Naive Majority Class baseline (73.4% accuracy, 0% recall) to benchmark model performance
- Trained three supervised learning models:
Logistic Regression (high interpretability, best recall),
Random Forest (captures nonlinear patterns, moderate recall),
Gradient Boosting (highest accuracy/AUC) 
- Evaluated models using accuracy, precision, recall, AUC, and training time; created comparison tables and confusion matrices
- Key findings:
    - Logistic Regression detected the most churners (recall = 0.811)
    - Gradient Boosting achieved the strongest overall accuracy (0.802)
    - Contract type, tenure, and billing variables emerged as the most important predictors
- Identified issues such as class imbalance, Random Forest overfitting, and difficulty predicting “silent churners,” informing the refinement plan for Sprint 4

## Repository
[GitHub Repository](https://github.com/achu1911/inst414-capstone)
