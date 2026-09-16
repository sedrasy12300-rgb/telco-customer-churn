📊 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on their personal information, subscribed services, contract,
and billing details.

The project also includes an interactive Streamlit web application that allows users to enter customer information and receive a churn prediction with an estimated churn probability.

🚀 Live Demo

Try the application:

🔗 Streamlit app link will be added after deployment.

🎯 Project Goal
The goal of this project is to build a complete machine learning workflow:
Data → Preprocessing → Model Training → Prediction → Interactive Web App
The application is designed to help identify customers who may be at risk of leaving the service.
📂 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains information about:
Customer demographics
Tenure
Phone and internet services
Additional services
Contract type
Payment method
Monthly charges
Total charges
Customer churn status
The customerID column was removed because it is an identifier and does not provide useful information for prediction.

🧹 Data Preprocessing

The following preprocessing steps were performed:
Removed customerID
Converted Churn from categorical values to binary values:
Yes → 1
No → 0
Converted TotalCharges to numeric values
Handled missing values in TotalCharges
Applied one-hot encoding to categorical features
Prepared the features and target variable for machine learning
🤖 Machine Learning Model

The project uses a Random Forest Classifier.
GridSearchCV was used for hyperparameter tuning.
Because identifying customers who may churn was an important goal, recall was used as the main optimization metric during model tuning.

The final model uses class balancing to help handle the imbalance between churned and non-churned customers.

📈 Model Evaluation
The model was evaluated using:
Accuracy
Recall
Confusion Matrix
Classification Report
Cross-validation
The model achieved approximately 73.9% accuracy on the test set.
The model was also optimized with a focus on detecting the positive churn class.
🖥️ Streamlit Web Application
The project includes an interactive Streamlit interface.
Users can enter three groups of information:
Customer Information
Gender
Senior Citizen
Partner
Dependents
Tenure
Services
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Billing & Contract
Contract
Paperless Billing
Payment Method
Monthly Charges
Total Charges
After clicking Predict Customer Churn, the application displays:
Whether the customer is likely to churn or stay
The estimated churn probability
📁 Project Structure

telco-customer-churn/
│
├── app.py
├── customer_churn_prediction.py
├── telco_churn_model.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md

🛠️ Technologies
Python
Pandas
Scikit-learn
Joblib
Streamlit
Matplotlib
GitHub

▶️ Run Locally
Install the required libraries:
pip install streamlit pandas scikit-learn joblib
Run the Streamlit application:
streamlit run app.py
💡 What I Learned
Through this project, I practiced:
Data preprocessing
Categorical feature encoding
Handling missing values
Random Forest classification
Hyperparameter tuning with GridSearchCV
Model evaluation using recall and accuracy
Saving and loading a trained model
Building an interactive ML application with Streamlit
Connecting a machine learning model to a user interface
👩‍💻 Author
Sedra Abdulhamid Marei
Computer Engineering Student

Machine Learning & Data Science
GitHub: sedrasy12300-rgb
