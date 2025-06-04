import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Basic info
print("🔍 Dataset Loaded")
print(df.info())
print("\n🎯 Class distribution before balancing:")
print(df['Class'].value_counts())

# Step 3: Check for nulls
if df.isnull().sum().any():
    print("\n⚠️ Missing values found:")
    print(df.isnull().sum())
    # If needed, fill or drop — not needed for this dataset
else:
    print("\n✅ No missing values.")

# Step 4: Drop duplicates (if any)
duplicates = df.duplicated().sum()
print(f"\n🧾 Duplicate rows: {duplicates}")
df = df.drop_duplicates()

# Step 5: Balance the dataset (undersampling)
fraud_df = df[df['Class'] == 1]
non_fraud_df = df[df['Class'] == 0].sample(n=len(fraud_df), random_state=42)

balanced_df = pd.concat([fraud_df, non_fraud_df])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n🎯 Class distribution after balancing:")
print(balanced_df['Class'].value_counts())

# Step 6: Scale 'Amount' column
scaler = StandardScaler()
balanced_df['Amount'] = scaler.fit_transform(balanced_df[['Amount']])

# Step 7: Drop unneeded 'Time' column
if 'Time' in balanced_df.columns:
    balanced_df.drop(['Time'], axis=1, inplace=True)

# Step 8: Split features and labels
X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

# Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Final confirmation
print(f"\n✅ Data Preprocessing Complete!")
print(f"🔹 X_train: {X_train.shape}")
print(f"🔹 X_test : {X_test.shape}")
print(f"🔹 y_train: {y_train.shape}")
print(f"🔹 y_test : {y_test.shape}")

# Step 10: Save preprocessed data to CSVs (optional but good practice)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("\n💾 Preprocessed datasets saved as CSV.")