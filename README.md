# Student Performance Score Predictor 🎓📊

A complete, end-to-end Data Science and Machine Learning project that predicts a student's math performance score based on demographic and historical test indicators. 

This project demonstrates a standard data science project lifecycle:
1. **Synthetic Data Generation** with realistic statistical relationships.
2. **Exploratory Model Pipeline** that processes features and trains/saves the best regression model (Gradient Boosting/Random Forest).
3. **Interactive Web Application** built with Flask and a custom premium CSS theme (featuring glassmorphism, responsive controls, visual prediction gauge, and dynamic AI-generated insights).

---

## 📁 Project Structure

```text
student-performance-predictor/
├── data/
│   └── student_performance.csv     # Generated dataset (1,000 samples)
├── models/
│   ├── model.pkl                   # Best serialized ML pipeline
│   └── metadata.pkl                # Metadata (features, metrics)
├── static/
│   └── style.css                   # Custom glassmorphic CSS styling
├── templates/
│   └── index.html                  # Flask HTML template for UI
├── app.py                          # Flask web application & Prediction API
├── generate_data.py                # Synthetic dataset generator script
├── train.py                        # Model training and validation script
├── requirements.txt                # Python dependencies list
└── README.md                       # Setup & usage instructions
```

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python 3.8+
* **Data Processing & Analytics:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, Standard Scaler, One Hot Encoder, Gradient Boosting Regressor, Random Forest Regressor)
* **Model Serialization:** Joblib
* **Server Framework:** Flask
* **Frontend Design:** Semantic HTML5, Vanilla CSS3 (Custom design system, CSS Grid/Flexbox, dynamic animations, HSL palette)

---

## 🚀 Getting Started (Setup & Execution)

Follow these steps to run the student performance predictor on your local machine:

### 1. Set Up Your Workspace
Open this folder in your terminal or IDE:
```bash
cd C:\Users\vikra\.gemini\antigravity-ide\scratch\student-performance-predictor
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

### 3. Generate the Dataset
Create the simulated student performance dataset by executing the generation script:
```bash
python generate_data.py
```
This generates `data/student_performance.csv` containing 1,000 student records with realistic correlations (e.g. test preparation, parental education levels, and reading/writing capabilities impacting math performance scores).

### 4. Train the ML Model
Train the scikit-learn models, evaluate metrics, and save the best model:
```bash
python train.py
```
This script pre-processes the data using a `ColumnTransformer` (standardizing numeric scores and one-hot encoding categories) and trains a `Gradient Boosting Regressor` and a `Random Forest Regressor`. It saves the pipeline with the highest $R^2$ score inside `models/model.pkl`.

### 5. Launch the Web App
Run the Flask server:
```bash
python app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📊 How It Works (Data Features & Insights)

When you make predictions using the Web UI, the model processes the following feature columns:
* **Gender:** Male or Female (historically, gender profiles show minor variances across subjects).
* **Race/Ethnicity:** Standard groupings (Group A - E).
* **Parental Level of Education:** Ranging from "Some High School" to "Master's Degree".
* **Lunch Type:** Standard or Free/Reduced (acting as an economic indicator).
* **Test Preparation Course:** None or Completed (demonstrating structured exam preparation).
* **Reading & Writing Scores:** Numerical score inputs (0 to 100).

The interface uses standard ranges to output **EduPredict AI** ratings:
* **Outstanding Performance (Score >= 85):** Highlighted in Mint Teal.
* **Above Average (70 - 84):** Highlighted in Indigo Violet.
* **Average Performance (50 - 69):** Highlighted in Slate Grey.
* **Requires Assistance (Score < 50):** Highlighted in Crimson Pink.
