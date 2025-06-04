import pandas as pd
import joblib

# Step 1: Load the preprocessed test data
X_test = pd.read_csv("X_test.csv")

# Step 2: Load the saved Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Step 3: Make predictions
predictions = rf_model.predict(X_test)

# Step 4: Show first 10 predictions
print("🔍 First 10 predictions (0 = Not Fraud, 1 = Fraud):")
print(predictions[:10])

# Step 5: Show total frauds detected
total_frauds = sum(predictions)
print(f"\n🚨 Total fraudulent transactions detected in test set: {total_frauds}")