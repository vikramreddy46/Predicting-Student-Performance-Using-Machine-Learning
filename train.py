import os
import time
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
    best_metrics = {}
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time = (time.time() - start_time) / len(X_test)  # average prediction time per sample in seconds
        
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
            best_metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'pred_time_ms': float(pred_time * 1000)  # in milliseconds
            }
            
    print(f"\nBest Model: {best_model_name} with R2 Score of {best_r2:.4f}")
    
    # Save the best pipeline
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/model.pkl'
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved the best trained model pipeline to {model_save_path}!")
    
    # Calculate Training Accuracy (R^2 on training set)
    y_train_pred = best_pipeline.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    best_metrics['r2_train'] = float(r2_train)
    
    # Compile additional statistics for visualization
    # 1. Feature Importances
    regressor = best_pipeline.named_steps['regressor']
    importances = regressor.feature_importances_
    
    # Standard numerical order: Hours Studied, Previous Scores, Sleep Hours, Sample Papers, Attendance.
    # Extracurricular is OHE, which splits into two columns at index 5 and 6.
    importance_map = {
        'Previous Score': float(importances[1]),
        'Study Hours': float(importances[0]),
        'Attendance': float(importances[4]),
        'Practice Papers': float(importances[3]),
        'Sleep Hours': float(importances[2]),
        'Extracurricular Activities': float(importances[5] + importances[6]) if len(importances) > 6 else float(importances[5])
    }
    
    # 2. Scatter Plot Data (First 150 test samples)
    y_test_pred = best_pipeline.predict(X_test)
    scatter_data = []
    for act, pred in zip(y_test[:150], y_test_pred[:150]):
        scatter_data.append({'x': float(act), 'y': float(pred)})
        
    # 3. Residual Error Data
    residual_data = []
    for act, pred in zip(y_test[:150], y_test_pred[:150]):
        residual_data.append({'x': float(pred), 'y': float(act - pred)})
        
    # 4. Correlation Matrix of original numerical columns + target
    corr_df = df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Attendance', 'Performance Index']].corr()
    corr_data = {
        'labels': ['Study Hours', 'Previous Score', 'Sleep Hours', 'Practice Papers', 'Attendance', 'Performance Index'],
        'matrix': corr_df.values.tolist()
    }
    
    # Compile metadata card info
    metadata = {
        'model_name': best_model_name,
        'metrics': best_metrics,
        'feature_importances': importance_map,
        'scatter_data': scatter_data,
        'residual_data': residual_data,
        'correlation_data': corr_data,
        'dataset_records': len(df),
        'dataset_features': len(X.columns),
        'train_test_split': '80% Train / 20% Test',
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols
    }
    
    metadata_save_path = 'models/metadata.pkl'
    joblib.dump(metadata, metadata_save_path)
    print("Saved model metadata successfully!")

if __name__ == '__main__':
    train_model()
