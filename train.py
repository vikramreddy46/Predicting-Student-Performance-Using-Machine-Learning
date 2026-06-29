import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model():
    data_path = 'data/student_performance.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define features and target
    X = df.drop(columns=['Performance Index'])
    y = df['Performance Index']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Identify numerical and categorical columns
    categorical_cols = ['Extracurricular Activities']
    numerical_cols = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Attendance']
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Try different models and find the best one
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    best_r2 = -1
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel: {name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with R2 Score of {best_r2:.4f}")
    
    # Save the best pipeline
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/model.pkl'
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved the best trained model pipeline to {model_save_path}!")
    
    # Save feature names / preprocessor info for reference in app
    metadata = {
        'model_name': best_model_name,
        'r2_score': best_r2,
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    print("Saved model metadata.")

if __name__ == '__main__':
    train_model()
