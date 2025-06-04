# 💳 Fraud Detection Using Machine Learning

This project is a machine learning-based solution to detect credit card fraud using classification models such as Logistic Regression and Random Forest. It includes data preprocessing, model training, evaluation, and a simple prediction interface using Streamlit.



## 🚀 Features

- Preprocessed and cleaned credit card transaction data
- Trained models: Logistic Regression and Random Forest
- Model evaluation (accuracy, confusion matrix, ROC AUC)
- Streamlit web app to predict fraud from custom input
- Modular code structure (preprocess, predict, model)
- Deployment-ready setup



## 📁 Project Structure


fraud_detection/
├── app/
│   └── app.py                     # Streamlit UI code
│
├── model/
│   └── test_model.py              # Model evaluation (Confusion Matrix, ROC, etc.)
│
├── predict/
│   └── predict.py                 # Prediction from input or uploaded file
│
├── preprocess/
│   └── preprocess.py              # Data cleaning, scaling, splitting
│
├── __pycache__/                  # Auto-generated (can be ignored)
│
├── creditcard.csv                # Main dataset (large - skip upload to GitHub if size > 100MB)
├── X_train.csv                   # Optional - generated train/test sets
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── logistic_model.pkl            # Saved Logistic Regression model
├── random_forest_model.pkl       # Saved Random Forest model
│
├── requirements.txt              # List of Python packages
├── README.md                     # Project info
└── .gitignore                    # (Optional) ignore files like __pycache__, venv, etc.


## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt




📊 Dataset

The dataset used is the Credit Card Fraud Detection dataset from Kaggle.

> Note: The original dataset is large and not included in this repo.
You can download it from Kaggle and place it as creditcard.csv in your project folder.



Or use the smaller version:

import pandas as pd

df = pd.read_csv("creditcard_sample.csv")  # Included for demo purposes




🧠 How It Works

1. Preprocess data (cleaning, splitting, scaling)


2. Train models (Logistic Regression, Random Forest)


3. Evaluate (Accuracy, Confusion Matrix, ROC)


4. Make predictions from custom input or file


5. Run Streamlit app for interactive use






💻 Run the App Locally

streamlit run app/app.py




🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub


2. Go to streamlit.io


3. Click Deploy → Select your repo


4. Set the main file path to app/app.py


5. Done 🎉






🙋‍♀️ Author

Yogitha, Aarna, Bhuvan 
CSE-B
Matrusri Engineering College.


