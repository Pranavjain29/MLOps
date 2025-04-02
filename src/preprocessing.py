import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

def transform_data(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a new DataFrame with scaled features
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df['target'] = y
    
    # Save the transformed data
    scaled_df.to_csv(output_file, index=False)
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transform_data.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    transform_data(input_file, output_file)