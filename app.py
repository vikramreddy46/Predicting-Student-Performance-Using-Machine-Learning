import os
import time
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths to models and DB
MODEL_PATH = 'models/model.pkl'
METADATA_PATH = 'models/metadata.pkl'
DATABASE_PATH = 'data/predictions.db'

model_pipeline = None
model_metadata = None

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            hours_studied REAL,
            previous_score REAL,
            sleep_hours REAL,
            papers_practiced REAL,
            attendance REAL,
            extracurricular TEXT,
            predicted_score REAL,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

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

# Initialize DB and load model at startup
init_db()
load_model()

@app.route('/')
def home():
    model_loaded = (model_pipeline is not None)
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/metadata')
def get_metadata():
    global model_metadata
    if model_metadata is None:
        load_model()
    if model_metadata is not None:
        return jsonify(model_metadata)
    else:
        return jsonify({'error': 'Metadata not available'}), 404

@app.route('/history')
def get_history():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, previous_score, attendance, hours_studied, 
                   sleep_hours, papers_practiced, extracurricular, predicted_score, category 
            FROM predictions ORDER BY id DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                'date': r[0],
                'previous_score': r[1],
                'attendance': r[2],
                'hours_studied': r[3],
                'sleep_hours': r[4],
                'papers_practiced': r[5],
                'extracurricular': r[6],
                'predicted_score': r[7],
                'category': r[8]
            })
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_csv')
def export_csv():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query('''
            SELECT timestamp as Date,
                   previous_score as "Previous Score",
                   attendance as "Attendance (%)",
                   hours_studied as "Study Hours/Day",
                   sleep_hours as "Sleep Hours/Night",
                   papers_practiced as "Sample Papers Solved",
                   extracurricular as "Extracurricular Activities",
                   predicted_score as "Predicted Score",
                   category as "Performance Category"
            FROM predictions ORDER BY id DESC
        ''', conn)
        conn.close()
        
        csv_data = df.to_csv(index=False)
        return csv_data, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=edupredict_history.csv'
        }
    except Exception as e:
        return str(e), 500

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
            
        # Parse inputs matching the sliders defined
        try:
            previous_scores = float(data.get('previous_scores', 0))
            attendance = float(data.get('attendance', 0))
            hours_studied = float(data.get('hours_studied', 0))
            sleep_hours = float(data.get('sleep_hours', 0))
            sample_papers_practiced = float(data.get('sample_papers_practiced', 0))
            extracurricular_activities = data.get('extracurricular_activities', 'No')
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Numeric fields must be valid numbers.'
            }), 400
            
        # Validate ranges matching the sliders required
        if not (0 <= previous_scores <= 100):
            return jsonify({'success': False, 'error': 'Previous Exam Score must be between 0 and 100.'}), 400
        if not (0 <= attendance <= 100):
            return jsonify({'success': False, 'error': 'Attendance must be between 0 and 100.'}), 400
        if not (0 <= hours_studied <= 12):
            return jsonify({'success': False, 'error': 'Study Hours must be between 0 and 12.'}), 400
        if not (4 <= sleep_hours <= 10):
            return jsonify({'success': False, 'error': 'Sleep Hours must be between 4 and 10.'}), 400
        if not (0 <= sample_papers_practiced <= 20):
            return jsonify({'success': False, 'error': 'Sample Question Papers must be between 0 and 20.'}), 400
            
        # Build pandas DataFrame for prediction (must match the training feature schema)
        input_data = pd.DataFrame([{
            'Hours Studied': hours_studied,
            'Previous Scores': previous_scores,
            'Extracurricular Activities': extracurricular_activities,
            'Sleep Hours': sleep_hours,
            'Sample Question Papers Practiced': sample_papers_practiced,
            'Attendance': attendance
        }])
        
        # Predict Performance Index & record time
        start_time = time.time()
        predicted_score = model_pipeline.predict(input_data)[0]
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Cap prediction between 10 and 100
        predicted_score = max(10.0, min(100.0, round(float(predicted_score), 2)))
        
        # Determine performance category
        if predicted_score >= 90:
            category = 'Excellent'
        elif predicted_score >= 80:
            category = 'Very Good'
        elif predicted_score >= 70:
            category = 'Good'
        elif predicted_score >= 50:
            category = 'Average'
        else:
            category = 'Needs Improvement'
            
        # Generate personalized recommendations
        recommendations = []
        if attendance < 75:
            recommendations.append("Improve attendance to increase your predicted score.")
        if hours_studied < 5:
            recommendations.append("Spend at least 6 hours studying every day.")
        if sleep_hours < 6:
            recommendations.append("Maintain 7–8 hours of sleep for better academic performance.")
        if sample_papers_practiced < 4:
            recommendations.append("Solve more sample question papers to build exam confidence.")
        if extracurricular_activities == 'No':
            recommendations.append("Consider participating in extracurricular activities to develop general skills.")
        if not recommendations:
            recommendations.append("Excellent study patterns! Keep up the consistent efforts.")
            
        # Save prediction in SQLite database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (hours_studied, previous_score, sleep_hours, papers_practiced, attendance, extracurricular, predicted_score, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (hours_studied, previous_scores, sleep_hours, sample_papers_practiced, attendance, extracurricular_activities, predicted_score, category))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'predicted_score': predicted_score,
            'category': category,
            'prediction_time_ms': round(prediction_time_ms, 2),
            'recommendations': recommendations,
            'inputs': {
                'previous_scores': previous_scores,
                'attendance': attendance,
                'hours_studied': hours_studied,
                'sleep_hours': sleep_hours,
                'sample_papers_practiced': sample_papers_practiced,
                'extracurricular_activities': extracurricular_activities
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
