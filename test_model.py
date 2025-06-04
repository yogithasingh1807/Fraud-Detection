import pandas as pd
import joblib

# Load your original dataset
df = pd.read_csv('creditcard.csv')

# Take 1 fraud and 1 non-fraud
sample = pd.concat([
    df[df['Class'] == 1].sample(1, random_state=1),  # fraud
    df[df['Class'] == 0].sample(1, random_state=1)   # not fraud
])

# Save original labels to compare
true_labels = sample['Class'].values

# Drop label + time column
sample = sample.drop(columns=['Class', 'Time'])

# Load scaler and scale 'Amount'
scaler = joblib.load('scaler.pkl')
sample['Amount'] = scaler.transform(sample[['Amount']])

# Load model and predict
model = joblib.load('fraud_detection_model.pkl')
predicted = model.predict(sample)

# Show results
print("\n✅ True Labels:", true_labels)
print("🧠 Model Predictions:", predicted)