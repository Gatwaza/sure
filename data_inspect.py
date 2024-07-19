import pandas as pd

dataset_path = '/Users/kigali/Desktop/sure/Heart_Disease_Prediction.csv'
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())
