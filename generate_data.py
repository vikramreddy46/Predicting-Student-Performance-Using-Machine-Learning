import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def generate_student_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Base seed data provided by user (20 samples)
    seed_data = [
        [7, 85, 'Yes', 7, 5, 92, 88],
        [5, 78, 'No', 6, 3, 85, 75],
        [8, 90, 'Yes', 8, 6, 96, 94],
        [6, 82, 'No', 7, 4, 88, 81],
        [9, 95, 'Yes', 8, 8, 98, 98],
        [4, 70, 'No', 6, 2, 80, 68],
        [3, 65, 'No', 5, 1, 75, 60],
        [10, 98, 'Yes', 8, 10, 99, 100],
        [2, 55, 'No', 5, 0, 70, 50],
        [7, 88, 'Yes', 7, 6, 93, 90],
        [6, 80, 'No', 6, 4, 86, 79],
        [5, 76, 'Yes', 7, 3, 84, 74],
        [8, 92, 'Yes', 8, 7, 97, 95],
        [4, 68, 'No', 6, 2, 79, 65],
        [9, 96, 'Yes', 8, 9, 99, 99],
        [3, 60, 'No', 5, 1, 72, 55],
        [7, 86, 'Yes', 7, 5, 91, 87],
        [5, 74, 'No', 6, 3, 82, 72],
        [8, 91, 'Yes', 8, 7, 96, 93],
        [6, 79, 'No', 7, 4, 87, 78]
    ]
    
    columns = [
        'Hours Studied', 
        'Previous Scores', 
        'Extracurricular Activities', 
        'Sleep Hours', 
        'Sample Question Papers Practiced', 
        'Attendance', 
        'Performance Index'
    ]
    
    df_seed = pd.DataFrame(seed_data, columns=columns)
    
    # 2. Fit a Linear Regression model on seed data to learn the statistical relationship
    # Map Extracurricular Activities to 1 (Yes) and 0 (No)
    X_seed = df_seed.drop(columns=['Performance Index']).copy()
    X_seed['Extracurricular Activities'] = X_seed['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    y_seed = df_seed['Performance Index']
    
    lr = LinearRegression()
    lr.fit(X_seed, y_seed)
    
    print("Fitted Linear Regression on seed data to learn coefficients:")
    for col, coef in zip(X_seed.columns, lr.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  Intercept: {lr.intercept_:.4f}")
    
    # 3. Generate remaining synthetic samples
    n_synthetic = n_samples - len(seed_data)
    
    # Simulating realistic distributions based on seed data
    hours = np.random.randint(1, 11, size=n_synthetic)
    prev_scores = np.clip(np.random.normal(loc=78, scale=12, size=n_synthetic).astype(int), 40, 100)
    extra_curr = np.random.choice(['Yes', 'No'], size=n_synthetic, p=[0.5, 0.5])
    sleep = np.random.randint(4, 10, size=n_synthetic)
    papers = np.random.randint(0, 11, size=n_synthetic)
    attendance = np.clip(np.random.normal(loc=85, scale=8, size=n_synthetic).astype(int), 60, 100)
    
    df_synthetic = pd.DataFrame({
        'Hours Studied': hours,
        'Previous Scores': prev_scores,
        'Extracurricular Activities': extra_curr,
        'Sleep Hours': sleep,
        'Sample Question Papers Practiced': papers,
        'Attendance': attendance
    })
    
    # Map Extracurricular for prediction
    X_synthetic = df_synthetic.copy()
    X_synthetic['Extracurricular Activities'] = X_synthetic['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    
    # Predict using the learned linear relationship + small normal noise
    preds = lr.predict(X_synthetic)
    noise = np.random.normal(0, 0.8, size=n_synthetic)
    perf_index = np.clip(np.round(preds + noise).astype(int), 10, 100)
    
    df_synthetic['Performance Index'] = perf_index
    
    # Combine seed and synthetic data
    df_final = pd.concat([df_seed, df_synthetic], ignore_index=True)
    return df_final

if __name__ == '__main__':
    print("Generating student performance dataset based on user seed data...")
    df = generate_student_data(n_samples=1000)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/student_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated and saved to {output_path}!")
    print(f"Total samples: {len(df)}")
    print(df.head())
    print("\nDataset description:")
    print(df.describe())
