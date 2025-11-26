# Credit Card Fraud Detection Using Machine Learning
This project develops a machine learning–based system to detect fraudulent credit card transactions using the public Kaggle “Credit Card Fraud Detection” dataset which contains anonymized PCA transformed features for over 280,000 European transactions with only 492 fraud cases (0.172%). Because the data is extremely imbalanced, several resampling strategies, namely oversampling, undersampling, and SMOTE, were tested to improve minority‐class detection.
After evaluating multiple models, including Logistic Regression, Decision Tree, Random Forest, and XGBoost, the combination of XGBoost with Random Oversampling delivered the best performance with an F1-score of 0.8913 and ROC-AUC of 0.9853.
A Streamlit web application was developed to demonstrate real-time prediction using the trained model, with automatic preprocessing and user-friendly visualization tools. The final system provides an interpretable, accessible, and efficient fraud detection workflow.

## Features
- Exploratory data analysis and visualization
- Preprocessing pipeline: scaling + PowerTransformer
- Balancing techniques:
  Random Oversampling: reduced majority class size, but caused information loss. 
  Random Undersampling: duplicated minority samples, improving recall but introducing redundancy. 
  SMOTE: generated synthetic samples using feature-space similarities, leading to better diversity and generalization. 
- Evaluation of multiple ML models:
  Logistic Regression: Served as baseline linear classifier 
  Decision Tree: Captured non-linear and rule-based relationships 
  Random Forest: Leveraged ensemble learning to improve robustness and reduce overfitting. 
  XGBoost: A gradient boosting framework designed for speed, performance and handling imbalanced data efficiently. 
- Final model: XGBoost with high F1-score and ROC-AUC
- Streamlit app for real-time fraud prediction

## Project Structure
.
├── app.py
├── credit.ipynb
├── Models/
│   └── xgboost_fraud_model_oversampled.pkl
├── Dataset/
│   └── creditcard.csv
└── README.md

## Installation
pip install -r requirements.txt

## Running the Streamlit App
streamlit run app.py

## Example Usage
- Paste a full transaction row into the prediction page
- View parsed features, fraud probability, and visual outputs

## Technologies Used
- Python, pandas, numpy
- scikit-learn
- XGBoost
- imbalanced-learn
- Streamlit

