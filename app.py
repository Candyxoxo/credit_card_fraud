

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the credit card fraud dataset"""
    try:
        df = pd.read_csv('D:\\Dummy\\Credit_Card_Fraud\\Dataset\\creditcard.csv')
        return df
    except FileNotFoundError:
        st.error("creditcard.csv file not found. Please make sure it's in the same directory.")
        return None

@st.cache_resource
def load_model():
    """Load the saved model"""
    try:
        # Load the trained model
        model = pickle.load(open('D:\\Dummy\\Credit_Card_Fraud\\Models\\xgboost_fraud_model_oversampled.pkl', 'rb'))
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please make sure model file is in the specified directory.")
        return None

def create_preprocessors(df):
    """Create and fit preprocessing objects based on the dataset"""
    # Create preprocessing objects
    scaler = StandardScaler()
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    
    # Fit scaler on Amount feature
    scaler.fit(df[['Amount']])
    
    # Fit power transformer on all features (excluding Class and Time)
    feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                      'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                      'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    power_transformer.fit(df[feature_columns])
    
    return scaler, power_transformer

def show_model_performance():
    """Display model performance comparison section"""
    st.header("Model Performance Analysis")
    
    st.markdown("""
    This section compares different machine learning models trained with various balancing techniques 
    to handle the imbalanced fraud detection dataset.
    """)
    
    # Create tabs for different balancing techniques
    tab1, tab2, tab3, tab4 = st.tabs(["Overall Comparison", "Oversampling", "Undersampling", "SMOTE"])
    
    with tab1:
        st.subheader("Model Performance Across All Techniques")
        
        # Summary table
        summary_data = {
            'Technique': ['Oversampling', 'Undersampling', 'SMOTE'],
            'Best Model': ['XGBoost', 'Logistic Regression', 'XGBoost'],
            'Accuracy': [0.9996, 0.9742, 0.9995],
            'F1 Score': [0.8913, 0.1103, 0.8513],
            'ROC-AUC': [0.9853, 0.9855, 0.9851]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Style the dataframe
        st.dataframe(
            summary_df.style.highlight_max(subset=['Accuracy', 'F1 Score', 'ROC-AUC'], color='lightgreen'),
            use_container_width=True
        )
        
        # Key insights
        st.markdown("""
        ### Key Insights
        
        **1. Undersampling Analysis:**
        - Leads to loss of data and information as it reduces the majority class
        - Models performed decently with high ROC-AUC scores
        - However, F1-scores are much lower than Oversampling or SMOTE
        - **Not ideal** for this fraud detection use case
        
        **2. Oversampling and SMOTE:**
        - Both produced excellent ROC-AUC scores (>0.98)
        - Shows models can distinguish between the 2 classes very well
        - Maintains high accuracy while improving minority class detection
        
        **3. XGBoost Performance:**
        - Achieved the highest F1-scores (0.85 - 0.89) across techniques
        - Very high ROC-AUC (>0.98) indicating excellent precision-recall balance
        - Strong discrimination ability between fraud and non-fraud classes
        - **Selected as the production model** for this application
        """)
        
        # Visualization comparing techniques
        st.subheader("Technique Comparison Visualization")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = summary_df[metric].values
            techniques = summary_df['Technique'].values
            
            bars = ax.bar(techniques, values, color=colors, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Oversampling Results")
        
        st.markdown("""
        **Oversampling** duplicates minority class samples to balance the dataset.
        This technique is effective when you want to preserve all information from both classes.
        """)
        
        # Oversampling data
        oversample_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'XGBoost'],
            'Accuracy': [0.972894, 0.999140, 0.980228],
            'F1 Score': [0.104408, 0.746114, 0.867145],
            'ROC-AUC': [0.9802, 0.891304, 0.985263]
        }
        oversample_df = pd.DataFrame(oversample_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(
                oversample_df.style.highlight_max(subset=['Accuracy', 'F1 Score', 'ROC-AUC'], color='lightgreen'),
                use_container_width=True
            )
        
        with col2:
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(oversample_df['Model']))
            width = 0.25
            
            bars1 = ax.bar(x - width, oversample_df['Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x, oversample_df['F1 Score'], width, label='F1 Score', color='#e74c3c', alpha=0.8)
            bars3 = ax.bar(x + width, oversample_df['ROC-AUC'], width, label='ROC-AUC', color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison - Oversampling')
            ax.set_xticks(x)
            ax.set_xticklabels(oversample_df['Model'], rotation=15, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.success("**Best Model: XGBoost** - Highest F1 Score (0.8913) and ROC-AUC (0.9853)")
    
    with tab3:
        st.subheader("Undersampling Results")
        
        st.markdown("""
        **Undersampling** reduces majority class samples to match the minority class.
        While this creates a balanced dataset, it can lead to information loss.
        """)
        
        # Undersampling data
        undersample_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.974228, 0.965749, 0.971981, 0.962098],
            'F1 Score': [0.110303, 0.081017, 0.102362, 0.078532],
            'ROC-AUC': [0.985549, 0.920935, 0.979061, 0.983729]
        }
        undersample_df = pd.DataFrame(undersample_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(
                undersample_df.style.highlight_max(subset=['Accuracy', 'F1 Score', 'ROC-AUC'], color='lightgreen'),
                use_container_width=True
            )
        
        with col2:
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(undersample_df['Model']))
            width = 0.25
            
            bars1 = ax.bar(x - width, undersample_df['Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x, undersample_df['F1 Score'], width, label='F1 Score', color='#e74c3c', alpha=0.8)
            bars3 = ax.bar(x + width, undersample_df['ROC-AUC'], width, label='ROC-AUC', color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison - Undersampling')
            ax.set_xticks(x)
            ax.set_xticklabels(undersample_df['Model'], rotation=15, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.warning("**Note:** All models show significantly lower F1 scores due to information loss from undersampling")
    
    with tab4:
        st.subheader("SMOTE Results")
        
        st.markdown("""
        **SMOTE (Synthetic Minority Oversampling Technique)** creates synthetic examples to balance the dataset.
        It effectively improves the model's ability to learn from minority (fraud) cases without losing information from the majority class.
        
        However, it can sometimes introduce borderline or noisy samples.
        """)
        
        # SMOTE data
        smote_data = {
            'Model': ['Logistic Regression', 'Decision Tree','Random Forest', 'XGBoost'],
            'Accuracy': [0.970296, 0.997630, 0.971981, 0.999491],
            'F1 Score': [0.097118, 0.519573, 0.102362, 0.851282],
            'ROC-AUC': [0.978922, 0.871482, 0.979061, 0.985131]
        }
        smote_df = pd.DataFrame(smote_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(
                smote_df.style.highlight_max(subset=['Accuracy', 'F1 Score', 'ROC-AUC'], color='lightgreen'),
                use_container_width=True
            )
        
        with col2:
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(smote_df['Model']))
            width = 0.25
            
            bars1 = ax.bar(x - width, smote_df['Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x, smote_df['F1 Score'], width, label='F1 Score', color='#e74c3c', alpha=0.8)
            bars3 = ax.bar(x + width, smote_df['ROC-AUC'], width, label='ROC-AUC', color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison - SMOTE')
            ax.set_xticks(x)
            ax.set_xticklabels(smote_df['Model'], rotation=15, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.success("**Best Model: XGBoost** - Highest F1 Score (0.8513) and ROC-AUC (0.9851)")
    
    # Final recommendation section
    st.markdown("---")
    st.subheader("Final Model Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Model", "XGBoost", delta="Best Performance")
    with col2:
        st.metric("Training Technique", "Oversampling", delta="F1: 0.8913")
    with col3:
        st.metric("ROC-AUC Score", "0.9853", delta="Excellent")
    
    st.info("""
    **Why XGBoost with Oversampling?**
    
    1. **Highest F1 Score (0.8913):** Best balance between precision and recall
    2. **Excellent ROC-AUC (0.9853):** Strong ability to distinguish between fraud and non-fraud
    3. **High Accuracy (0.9996):** Maintains overall classification accuracy
    4. **Robust to Imbalance:** Handles the imbalanced nature of fraud data effectively
    5. **No Information Loss:** Oversampling preserves all original data while balancing classes
    """)

def parse_row_input(row_string):
    """Parse the pasted row string into a dictionary of feature values"""
    try:
        # Split the row by commas
        values = [x.strip().strip('"') for x in row_string.split(',')]
        
        # Expected columns in order: Time, V1-V28, Amount, Class
        expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
        
        if len(values) != len(expected_columns):
            st.error(f"Expected {len(expected_columns)} values, but got {len(values)}. Please check your input.")
            return None
        
        # Create feature dictionary
        feature_dict = {}
        for col, val in zip(expected_columns, values):
            try:
                feature_dict[col] = float(val)
            except ValueError:
                feature_dict[col] = val  # Keep as string if conversion fails
        
        return feature_dict
        
    except Exception as e:
        st.error(f"Error parsing row: {str(e)}")
        return None

def preprocess_user_input(feature_dict, scaler, power_transformer):
    """Preprocess user input using the same preprocessing as training"""
    # Remove Class and Time from features for prediction
    prediction_features = {k: v for k, v in feature_dict.items() if k not in ['Class', 'Time']}
    
    # Create DataFrame with correct column order (excluding Time)
    columns_order = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    input_df = pd.DataFrame([prediction_features])[columns_order]
    
    # Scale the Amount feature
    input_df['Amount'] = scaler.transform(input_df[['Amount']])
    
    # Apply power transformation to all features
    input_df_transformed = power_transformer.transform(input_df)
    
    return input_df_transformed, feature_dict.get('Class', None)

def show_class_distribution(df):
    """Display class distribution charts"""
    st.subheader("Class Distribution")
    
    # Calculate percentages
    classes = df['Class'].value_counts()
    percentage_normal = (classes[0]/len(df))*100
    percentage_fraud = (classes[1]/len(df))*100
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar plot 
        plt.figure(figsize=(6,4))
        sns.countplot(x='Class', data=df)
        plt.title('Class Distribution')
        plt.xticks([0, 1], ['Normal', 'Fraudulent'])
        plt.ylabel('Number of Transactions')
        st.pyplot(plt)
    
    with col2:
        # Pie chart 
        labels = ['Normal', 'Fraudulent']
        sizes = [percentage_normal, percentage_fraud]
        colors = ['#66b3ff','#ff9999']
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
        plt.title('Percentage Distribution of Transactions')
        st.pyplot(plt)
    
    # Display percentages as metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Transactions", f"{classes[0]:,}", f"{percentage_normal:.2f}%")
    with col2:
        st.metric("Fraudulent Transactions", f"{classes[1]:,}", f"{percentage_fraud:.2f}%")

def show_amount_distribution(df):
    """Display transaction amount distribution by class"""
    st.subheader("Transaction Amount Distribution by Class")
    
    plt.figure(figsize=(14,6))

    # Non-Fraud Plot
    plt.subplot(1,2,1)
    sns.histplot(df[df['Class']==0]['Amount'],
                 bins=50,
                 kde=True,
                 stat='density',
                 color='blue',
                 alpha=0.6)
    plt.xscale('log')
    plt.title("Non-Fraud Transactions")
    plt.xlabel("Transaction Amount (Log Scale)")
    plt.ylabel("Density")

    # Fraud Plot
    plt.subplot(1,2,2)
    sns.histplot(df[df['Class']==1]['Amount'],
                 bins=50,
                 kde=True,
                 stat='density',
                 color='red',
                 alpha=0.7)
    plt.xscale('log')
    plt.title("Fraud Transactions")
    plt.xlabel("Transaction Amount (Log Scale)")
    plt.ylabel("Density")

    plt.tight_layout()
    st.pyplot(plt)

def create_row_input_section():
    """Create input section for pasting complete rows"""
    st.header("Paste Complete Transaction Row for Fraud Prediction")
    
    st.markdown("""
    ### How to use:
    1. Copy a complete row from your dataset (including all 31 columns: Time, V1-V28, Amount, Class)
    2. Paste it in the text area below
    3. Click 'Predict Fraud' to get the classification result
    
    **Note:** The Time feature will be automatically removed before prediction as it's not used by the model.
    
    **Example row format:**
    ```
    0,1.191857,0.266151,0.166480,0.448154,0.060018,-0.082361,-0.078803,0.085102,-0.255425,-0.166974,1.612727,1.065235,0.489095,-0.143772,0.635558,0.463917,-0.114805,-0.183361,-0.145783,-0.069083,-0.225775,-0.638672,0.101288,-0.339846,0.167170,0.125895,-0.008983,0.014724,2.69,0
    ```
    """)
    
    with st.form("row_input_form"):
        row_input = st.text_area(
            "Paste complete transaction row:",
            value="0,1.191857,0.266151,0.166480,0.448154,0.060018,-0.082361,-0.078803,0.085102,-0.255425,-0.166974,1.612727,1.065235,0.489095,-0.143772,0.635558,0.463917,-0.114805,-0.183361,-0.145783,-0.069083,-0.225775,-0.638672,0.101288,-0.339846,0.167170,0.125895,-0.008983,0.014724,2.69,0",
            height=100,
            help="Paste a complete row with all 31 values separated by commas"
        )
        
        submitted = st.form_submit_button("Predict Fraud")
    
    return row_input, submitted

def display_feature_table(feature_dict):
    """Display the parsed features in a nice table format"""
    st.subheader("Parsed Transaction Features")
    
    # Create DataFrame for better display
    features_df = pd.DataFrame.from_dict(feature_dict, orient='index', columns=['Value'])
    
    # Show Time feature but indicate it will be removed
    if 'Time' in features_df.index:
        st.info(f"**Time:** {features_df.loc['Time', 'Value']:.6f} *(will be removed for prediction)*")
    
    # Remove Class and Time for cleaner display
    display_features = features_df.drop(['Class', 'Time'], errors='ignore')
    
    # Display in columns for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**V1-V10**")
        for i in range(1, 11):
            feature = f'V{i}'
            if feature in display_features.index:
                st.write(f"**{feature}:** {display_features.loc[feature, 'Value']:.6f}")
    
    with col2:
        st.write("**V11-V20**")
        for i in range(11, 21):
            feature = f'V{i}'
            if feature in display_features.index:
                st.write(f"**{feature}:** {display_features.loc[feature, 'Value']:.6f}")
    
    with col3:
        st.write("**V21-V28 & Amount**")
        for i in range(21, 29):
            feature = f'V{i}'
            if feature in display_features.index:
                st.write(f"**{feature}:** {display_features.loc[feature, 'Value']:.6f}")
        if 'Amount' in display_features.index:
            st.write(f"**Amount:** {display_features.loc['Amount', 'Value']:.6f}")
    
    # Show actual class if available
    if 'Class' in feature_dict:
        actual_class = "Fraudulent" if feature_dict['Class'] == 1 else "Normal"
        st.info(f"**Actual Label in Dataset:** {actual_class} (Class: {feature_dict['Class']})")

def display_prediction_results(prediction, probability, actual_class=None):
    """Display prediction results in a user-friendly format"""
    st.header("Prediction Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("""
            **FRAUDULENT TRANSACTION DETECTED**
            
            **Action Required:** This transaction has been flagged as potentially fraudulent.
            Please review the transaction details and consider contacting the cardholder.
            """)
        else:
            st.success("""
            **LEGITIMATE TRANSACTION**
            
            **Status:** This transaction appears to be legitimate based on our fraud detection model.
            """)
    
    with col2:
        # Display probability scores
        st.metric("Fraud Probability", f"{probability[1]:.4f}")
        st.metric("Legitimate Probability", f"{probability[0]:.4f}")
        
        # Show accuracy if actual class is known
        if actual_class is not None:
            is_correct = (prediction == actual_class)
            if is_correct:
                st.success("Prediction matches actual label")
            else:
                st.error("Prediction differs from actual label")
    
    # Probability visualization
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['green', 'red']
    labels = ['Legitimate', 'Fraudulent']
    
    bars = ax.barh(labels, [probability[0], probability[1]], color=colors, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Transaction Classification Probabilities')
    
    # Add probability values on bars
    for bar, prob in zip(bars, [probability[0], probability[1]]):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.4f}', 
                ha='left', va='center', fontweight='bold')
    
    st.pyplot(fig)

def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("""
    This application uses machine learning to detect fraudulent credit card transactions.
    The system employs an **XGBoost model trained with Oversampling technique** for optimal fraud detection.
    
    **Note:** The Time feature is automatically excluded from predictions as transactions are independent of time.
    """)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create preprocessing objects based on the dataset (excluding Time)
    with st.spinner("Setting up preprocessing pipeline..."):
        scaler, power_transformer = create_preprocessors(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Section", 
                                   ["Data Overview", "Model Performance", "Fraud Prediction"])
    
    if app_mode == "Data Overview":
        st.header("Dataset Overview")
        
        # Show basic dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            normal_count = len(df[df['Class'] == 0])
            st.metric("Normal Transactions", f"{normal_count:,}")
        with col3:
            fraud_count = len(df[df['Class'] == 1])
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        with col4:
            st.metric("Fraud Rate", f"{(fraud_count/len(df)*100):.4f}%")
        
        # Display the charts
        show_class_distribution(df)
        show_amount_distribution(df)
        
        # Show sample data
        st.subheader("Sample Data (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show dataset information
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Columns and Data Types:**")
            st.write(df.dtypes)
        
        with col2:
            st.write("**Statistical Summary:**")
            st.dataframe(df.describe())
        
        # Provide example rows for copying
        st.subheader("Example Rows for Testing")
        
        # Get actual examples from the dataset
        normal_example = df[df['Class'] == 0].iloc[0]
        fraud_example = df[df['Class'] == 1].iloc[0] if len(df[df['Class'] == 1]) > 0 else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Normal Transaction Example:**")
            normal_row = ",".join([str(x) for x in normal_example.values])
            st.code(normal_row, language='text')
            st.write(f"**Class:** {normal_example['Class']}")
        
        with col2:
            if fraud_example is not None:
                st.write("**Fraudulent Transaction Example:**")
                fraud_row = ",".join([str(x) for x in fraud_example.values])
                st.code(fraud_row, language='text')
                st.write(f"**Class:** {fraud_example['Class']}")
            else:
                st.write("**No fraudulent transactions found in sample**")
        
        # Show preprocessing information
        with st.expander("Preprocessing Information"):
            st.write("""
            **Preprocessing Pipeline:**
            
            1. **Feature Selection:**
               - Time feature is excluded from model training and prediction
               - Only V1-V28 and Amount features are used
            
            2. **StandardScaler** - Applied to 'Amount' feature
               - Removes mean and scales to unit variance
               - Fitted on the entire dataset's Amount values
            
            3. **PowerTransformer (Yeo-Johnson)** - Applied to V1-V28 and Amount
               - Makes data more Gaussian-like
               - Handles both positive and negative values
               - Fitted on V1-V28 and Amount features from the dataset
            """)
            
    elif app_mode == "Model Performance":
        show_model_performance()
            
    elif app_mode == "Fraud Prediction":
        # Get user input
        row_input, submitted = create_row_input_section()
        
        if submitted and row_input.strip():
            # Parse the row input
            feature_dict = parse_row_input(row_input)
            
            if feature_dict is not None:
                # Display parsed features
                display_feature_table(feature_dict)
                
                # Preprocess and predict
                with st.spinner("Processing transaction and making prediction..."):
                    try:
                        # Preprocess user input (Time will be automatically removed)
                        processed_input, actual_class = preprocess_user_input(feature_dict, scaler, power_transformer)
                        
                        # Use the actual trained model for prediction
                        prediction = model.predict(processed_input)[0]
                        probability = model.predict_proba(processed_input)[0]
                        
                        # Display results
                        display_prediction_results(prediction, probability, actual_class)
                        
                        # Show preprocessing details
                        with st.expander("Preprocessing Details"):
                            st.write("""
                            **Preprocessing Steps Applied:**
                            
                            1. **Feature Selection:**
                               - Time feature removed (not used by model)
                               - Using only V1-V28 and Amount features
                            
                            2. **Standard Scaling** (Amount feature):
                               - Scaled using StandardScaler fitted on dataset
                               - Formula: (x - mean) / std
                            
                            3. **Power Transformation** (V1-V28 and Amount):
                               - Yeo-Johnson transformation
                               - Makes data more Gaussian-distributed
                               - Handles skewness and outliers
                            
                            4. **Model Inference**:
                               - Processed features fed to trained XGBoost model
                               - Model trained with Oversampling technique
                            """)
                            
                            # Show model type if available
                            if hasattr(model, '__class__'):
                                st.write(f"**Model Type:** {model.__class__.__name__}")
                            
                    except Exception as e:
                        st.error(f"Error processing prediction: {str(e)}")
                        st.info("Please check that the row format is correct.")
        
        # Quick prediction from dataset samples
        st.subheader("Quick Predictions from Dataset")
        st.write("Select a transaction from the dataset for instant prediction:")
        
        # Let user select a row from the dataset
        sample_size = min(50, len(df))
        sample_indices = st.selectbox("Select a transaction index:", range(sample_size))
        
        if st.button("Predict Selected Transaction"):
            selected_row = df.iloc[sample_indices]
            
            # Convert row to string format for display
            row_string = ",".join([str(x) for x in selected_row.values])
            
            # Parse and predict
            feature_dict = parse_row_input(row_string)
            if feature_dict is not None:
                display_feature_table(feature_dict)
                
                with st.spinner("Making prediction..."):
                    processed_input, actual_class = preprocess_user_input(feature_dict, scaler, power_transformer)
                    prediction = model.predict(processed_input)[0]
                    probability = model.predict_proba(processed_input)[0]
                    
                    display_prediction_results(prediction, probability, actual_class)

if __name__ == "__main__":
    main()