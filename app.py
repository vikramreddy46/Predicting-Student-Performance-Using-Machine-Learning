import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths to models
MODEL_PATH = 'models/model.pkl'
METADATA_PATH = 'models/metadata.pkl'

model_pipeline = None
model_metadata = None

def load_model():
    global model_pipeline, model_metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            model_metadata = joblib.load(METADATA_PATH)
            print("Model and metadata loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("Model file not found. Please train the model using train.py first.")
        return False

# Load model at startup
load_model()

@app.route('/')
def home():
    model_loaded = (model_pipeline is not None)
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    global model_pipeline
    if model_pipeline is None:
        if not load_model():
            return jsonify({
                'success': False,
                'error': 'Model not trained or loaded. Please run train.py first.'
            }), 500
            
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        # Parse inputs
        try:
            hours_studied = float(data.get('hours_studied', 0))
            previous_scores = float(data.get('previous_scores', 0))
            extracurricular_activities = data.get('extracurricular_activities', 'No')
            sleep_hours = float(data.get('sleep_hours', 0))
            sample_papers_practiced = float(data.get('sample_papers_practiced', 0))
            attendance = float(data.get('attendance', 0))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Numeric fields must be valid numbers.'
            }), 400
            
        # Validation checks
        if not (1 <= hours_studied <= 10):
            return jsonify({'success': False, 'error': 'Hours Studied must be between 1 and 10.'}), 400
        if not (0 <= previous_scores <= 100):
            return jsonify({'success': False, 'error': 'Previous Scores must be between 0 and 100.'}), 400
        if not (4 <= sleep_hours <= 10):
            return jsonify({'success': False, 'error': 'Sleep Hours must be between 4 and 10.'}), 400
        if not (0 <= sample_papers_practiced <= 10):
            return jsonify({'success': False, 'error': 'Sample Question Papers Practiced must be between 0 and 10.'}), 400
        if not (0 <= attendance <= 100):
            return jsonify({'success': False, 'error': 'Attendance must be between 0 and 100.'}), 400
            
        # Build pandas DataFrame for prediction (must match the training feature schema)
        input_data = pd.DataFrame([{
            'Hours Studied': hours_studied,
            'Previous Scores': previous_scores,
            'Extracurricular Activities': extracurricular_activities,
            'Sleep Hours': sleep_hours,
            'Sample Question Papers Practiced': sample_papers_practiced,
            'Attendance': attendance
        }])
        
        # Predict Performance Index
        predicted_index = model_pipeline.predict(input_data)[0]
        predicted_index = max(10.0, min(100.0, round(float(predicted_index), 2)))
        
        return jsonify({
            'success': True,
            'predicted_performance_index': predicted_index,
            'inputs': {
                'hours_studied': hours_studied,
                'previous_scores': previous_scores,
                'extracurricular_activities': extracurricular_activities,
                'sleep_hours': sleep_hours,
                'sample_papers_practiced': sample_papers_practiced,
                'attendance': attendance
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Flask Web Application...")
    app.run(debug=True, port=5000)
