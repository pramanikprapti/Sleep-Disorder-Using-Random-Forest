import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("sleep_data.csv")

# Handle 'Blood Pressure' column (e.g., '130/85')
if 'Blood Pressure' in data.columns:
    # Split 'Blood Pressure' into two columns: 'Systolic' and 'Diastolic'
    data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic'] = pd.to_numeric(data['Systolic'], errors='coerce')  # Convert to numeric
    data['Diastolic'] = pd.to_numeric(data['Diastolic'], errors='coerce')  # Convert to numeric
    data.drop(columns=['Blood Pressure'], inplace=True)  # Drop the original column



# Encode 'Gender' (Male=0, Female=1) using LabelEncoder
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encode 'BMI Category' (e.g., 'Normal' = 0, 'Overweight' = 1, 'Obese' = 2)
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])

# If there are other categorical columns (like 'Occupation'), you can also encode them
occupation_encoded = pd.get_dummies(data['Occupation'], drop_first=True)
data = pd.concat([data, occupation_encoded], axis=1)
data.drop(columns=["Occupation"], inplace=True)

# Now let's handle the target variable, 'Sleep Disorder', if necessary
# Assuming 'Sleep Disorder' is the target column and needs to be encoded too
data['Sleep Disorder'] = label_encoder.fit_transform(data['Sleep Disorder'].fillna('No Disorder'))

# Split the dataset into features (X) and target (y)
X = data.drop(columns=["Sleep Disorder"])  # Features
y = data["Sleep Disorder"]  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (you can change this model if needed)
model = RandomForestClassifier(random_state=50)
model.fit(X_train, y_train)

# Save the trained model and feature columns
joblib.dump(model, 'sleep_disorder_model.pkl')

# Save the feature column names for reference
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')

print("Model and columns saved successfully!")