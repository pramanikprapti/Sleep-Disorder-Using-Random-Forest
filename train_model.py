import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("sleep_data.csv")

# Print column names to check the correct target column name
print("Columns in dataset:", df.columns)

# Ensure the correct target column name
target_column = "Sleep Disorder"  # Corrected column name
if target_column not in df.columns:
    raise ValueError(f"Error: Target column '{target_column}' not found in dataset!")

# Drop 'Person ID' and separate features & target
X = df.drop(columns=["Person ID", target_column])
y = df[target_column]  # Target variable

# Handle missing values in target variable
y = y.dropna()  # Remove rows where target is missing

# Align X and y after dropping missing target values
X = X.loc[y.index]  

# Identify numeric columns
numeric_cols = X.select_dtypes(include=['number']).columns

# Handle missing values in features
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())  # Fill numeric columns with mean

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

#  Final check: Remove any remaining NaN rows 
X = X.dropna()
y = y.loc[X.index]  # Ensure target values match feature set

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # No more NaN errors

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

# Save model columns
with open("model_columns.pkl", "wb") as columns_file:
    pickle.dump(X.columns.tolist(), columns_file)

print("Model and columns saved successfully!")
