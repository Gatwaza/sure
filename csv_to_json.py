import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('/Users/kigali/Desktop/sure/Heart_Disease_Prediction.csv')

# Display the first few rows to understand the structure
print(df.head())

# Define a function to convert numerical and categorical data into a descriptive text
def create_textual_representation(row):
    # Example conversion of a row into a text description
    description = (
        f"Patient is a {row['Age']}-year-old "
        f"{'male' if row['Sex'] == 1 else 'female'} "
        f"with {'chest pain' if row['Chest pain type'] > 0 else 'no chest pain'}, "
        f"blood pressure of {row['BP']} mmHg, "
        f"cholesterol level of {row['Cholesterol']} mg/dL, "
        f"{'has' if row['FBS over 120'] == 1 else 'does not have'} fasting blood sugar above 120 mg/dL, "
        f"and {'exercise-induced angina' if row['Exercise angina'] == 1 else 'no exercise-induced angina'}. "
        f"Heart disease presence: {row['Heart Disease']}."
    )
    return description

# Apply the function to create text descriptions
df['text'] = df.apply(create_textual_representation, axis=1)

# Convert to JSON
json_data = df[['text', 'Heart Disease']].to_dict(orient='records')

# Save to a JSON file
json_file_path = '/Users/kigali/Desktop/sure/Heart_Disease_Prediction.json'
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print(f"Conversion to JSON completed. JSON saved at {json_file_path}.")
