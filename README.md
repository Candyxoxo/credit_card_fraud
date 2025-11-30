# **Credit Card Fraud Detection Using Machine Learning**

<div align="justify">

This project implements a complete machine learning pipeline to detect fraudulent credit card transactions using the public Kaggle "Credit Card Fraud Detection" dataset. The dataset contains anonymized, PCA-transformed numerical features extracted from over 284,000 European card transactions, of which only 492 (0.172%) are fraudulent — making this a highly imbalanced binary classification problem.

To address this imbalance and improve fraud detection, several preprocessing, transformation, resampling, and modeling strategies were applied. The final deployed model uses **XGBoost combined with Random Oversampling**, achieving strong predictive performance while maintaining computational efficiency and interpretability.

A **Streamlit web application** is also provided for real-time prediction and demonstration of how the trained model can be incorporated into a user-facing environment.

---

# **Problem Statement**

Credit card fraud is rare but highly consequential, causing billions in financial losses globally each year. Traditional ML models struggle to detect fraud due to:

1. **Extreme class imbalance** (fraud ≈ 0.17%)
2. **Lack of meaningful interpretability** in raw feature space
3. **Nonlinear and overlapping patterns** between fraud and non-fraud
4. **Distribution skewness** across PCA-transformed features
5. **High cost of false negatives**, where fraudulent transactions go undetected

**Objective:**
Build a robust machine learning system that:

* Handles severe imbalance
* Learns meaningful fraud patterns
* Provides accurate predictions
* Can be deployed and used for live inference

---

# **Dataset Details**

**Source:**
Kaggle — *Credit Card Fraud Detection*
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Size:**

* **284,807 total transactions**
* **492 fraud (0.172%)**
* **284,315 non-fraud**

**Features:**

* **V1–V28:** PCA-transformed components (numerical)
* **Amount:** original transaction amount
* **Time:** seconds elapsed between transactions (later dropped — no predictive value)
* **Class:** target label (0 = non-fraud, 1 = fraud)

The PCA transformation ensures anonymity and reduces the risk of linking back to personally identifiable information (PII).

---

# **Project Pipeline**

Below is the complete step-by-step workflow implemented in the notebook and app:

### **1. Exploratory Data Analysis**

* Distribution of fraud vs non-fraud
* Correlation inspection (heatmap)
* Amount and Time feature analysis
* Skewness visualization

### **2. Preprocessing**

* Dropped `Time` (correlation ≈ −0.012)
* Standardized `Amount` using `StandardScaler`
* Applied **PowerTransformer (Yeo-Johnson)** to reduce skewness
* Ensured consistent feature ordering

### **3. Handling Class Imbalance**

All three methods were tested:

| Method                   | Description                | Outcome                            |
| ------------------------ | -------------------------- | ---------------------------------- |
| **Random Oversampling**  | Duplicate minority samples | Best recall & F1                   |
| **Random Undersampling** | Remove majority samples    | Loss of information                |
| **SMOTE**                | Generate synthetic samples | Good generalization but some noise |

**Final choice: Random Oversampling**

### **4. Model Training & Evaluation**

Models evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* **XGBoost (final model)**

**Final Model Performance:**

* **F1-Score:** 0.8913
* **ROC-AUC:** 0.9853

### **5. Deployment (Streamlit App)**

* Loads trained model & preprocessors
* Accepts full transaction row from user
* Transforms input using saved pipeline
* Outputs:

  * **Fraud Prediction**
  * **Fraud Probability**
  * **Parsed Features**

---

# **Features Implemented**

### Data preprocessing pipeline

* Standard Scaling
* PowerTransformer
* PCA feature handling
* Removal of noisy features
* Skewness analysis

### Class imbalance correction

* Oversampling
* Undersampling
* SMOTE

### ML Models

* Logistic Regression
* Decision Tree
* Random Forest
* **XGBoost (best)**

### Streamlit Web App

* Real-time prediction
* Automatic preprocessing
* Probability outputs
* User-friendly interface

---

# **Model & Dataset Downloads**

### **Dataset (Kaggle)**

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Trained XGBoost Model (Google Drive)**

[https://drive.google.com/file/d/1p3mVl9Fo2FGPLt77M6AeTI471RBEWaAD/view?usp=drive_link](https://drive.google.com/file/d/1p3mVl9Fo2FGPLt77M6AeTI471RBEWaAD/view?usp=drive_link)


---

# **Project Structure**

```
.
├── app.py
├── credit.ipynb
├── Models/
│   └── xgboost_fraud_model_oversampled.pkl
├── Dataset/
│   └── creditcard.csv
└── README.md
```

---

# **Installation**

```bash
pip install -r requirements.txt
```

---

# **Running the Streamlit App**

```bash
streamlit run app.py
```

---

# **Example Usage**

1. Open the web app
2. Paste a full transaction row
3. View:

   * Parsed transaction features
   * Fraud / Not Fraud prediction
   * Probability score

---

# **Technologies Used**

* Python
* NumPy, Pandas
* scikit-learn
* XGBoost
* imbalanced-learn
* Streamlit

</div>

