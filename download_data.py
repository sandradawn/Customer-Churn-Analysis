import os
import urllib.request
import pandas as pd

# Define directories to create
directories = ["data", "notebooks", "models", "app"]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# URL of the dataset
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
output_path = os.path.join("data", "telco_churn.csv")

print(f"Downloading dataset from {url}...")
try:
    urllib.request.urlretrieve(url, output_path)
    print(f"Dataset downloaded successfully and saved to: {output_path}")
    
    # Load and verify
    df = pd.read_csv(output_path)
    print("\nVerification:")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Target distribution (Churn):")
    print(df['Churn'].value_counts(normalize=True))
except Exception as e:
    print(f"Failed to download dataset: {e}")
