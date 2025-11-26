
<div align="justify">

# <strong>Credit Card Fraud Detection Using Machine Learning</strong>

This project develops a machine learning–based system to detect fraudulent credit card transactions using the public Kaggle <strong>“Credit Card Fraud Detection”</strong> dataset, which contains anonymized PCA-transformed features for over <strong>280,000 European transactions</strong>, with only <strong>492 fraud cases (0.172%)</strong>.

Because the dataset is <strong>highly imbalanced</strong>, multiple sampling strategies—oversampling, undersampling, and SMOTE—were evaluated to improve minority-class detection.

After testing models such as <strong>Logistic Regression, Decision Tree, Random Forest, and XGBoost</strong>, the best performance was achieved using <strong>XGBoost with Random Oversampling</strong>, yielding:

* <strong>F1-Score:</strong> 0.8913
* <strong>ROC-AUC:</strong> 0.9853

A <strong>Streamlit web application</strong> was implemented to provide real-time prediction with automatic preprocessing and user-friendly visualizations. The final system is efficient, interpretable, and easy to deploy.



## <strong>Dataset & Model Downloads</strong>

* **Kaggle Dataset:**
  [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

* **Trained XGBoost Model (Google Drive):**
  [https://drive.google.com/file/d/1p3mVl9Fo2FGPLt77M6AeTI471RBEWaAD/view?usp=drive_link](https://drive.google.com/file/d/1p3mVl9Fo2FGPLt77M6AeTI471RBEWaAD/view?usp=drive_link)



## <strong>Features</strong>

* Exploratory data analysis and visualization
* Preprocessing pipeline: Standard Scaling and PowerTransformer
* Balancing techniques:

  * <strong>Random Oversampling:</strong> Duplicated minority samples, improving recall but causing redundancy
  * <strong>Random Undersampling:</strong> Reduced majority class size but led to information loss
  * <strong>SMOTE:</strong> Generated synthetic samples, improving diversity and generalization
* Evaluation of multiple machine learning models:

  * <strong>Logistic Regression:</strong> Baseline linear classifier
  * <strong>Decision Tree:</strong> Captures nonlinear, rule-based patterns
  * <strong>Random Forest:</strong> Ensemble method enhancing stability and reducing overfitting
  * <strong>XGBoost:</strong> Fast, powerful gradient boosting algorithm suited for imbalanced datasets
* Final best model: <strong>XGBoost with Random Oversampling</strong>
* Streamlit web app for real-time fraud prediction



## <strong>Project Structure</strong>

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



## <strong>Installation</strong>

```bash
pip install -r requirements.txt
```



## <strong>Running the Streamlit App</strong>

```bash
streamlit run app.py
```



## <strong>Example Usage</strong>

* Paste a full transaction row into the prediction UI
* View parsed features, predicted fraud probability, and visual outputs



## <strong>Technologies Used</strong>

* Python
* pandas, numpy
* scikit-learn
* XGBoost
* imbalanced-learn
* Streamlit

</div>

