# Workplace Fatigue Detection System

A machine learning project that predicts employee fatigue levels based on work patterns, sleep proxies, and behavioral signals.

This project simulates workplace data and builds a full ML pipeline including data generation, preprocessing, model training, evaluation, and prediction.

---

## Features

* Synthetic dataset generation (10,000 samples)
* Feature engineering based on real-world fatigue indicators
* Multi-class classification: **Low, Medium, High fatigue risk**
* Random Forest model with class balancing
* Evaluation using F1-score, confusion matrix, and feature importance
* Reusable prediction pipeline for new employee data

---

## 📂 Project Structure

```
├── data/
│   └── raw/
│       ├── fatigue_10k.csv
│       └── predict_input.csv
├── models/
│   ├── random_forest_fatigue.pkl
│   └── feature_columns.pkl
├── src/
│   ├── data_generation.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
```

---

## ⚙️ How It Works

### 1. Data Generation

Synthetic data is created using realistic assumptions:

* Longer working hours → higher fatigue
* More breaks → lower fatigue
* Poor sleep → higher fatigue
* Night shifts → increased fatigue impact

---

### 2. Preprocessing

* Converts fatigue score → categorical risk levels
* One-hot encodes `time_of_day`
* Standardizes numerical features

---

### 3. Model Training

* Model: `RandomForestClassifier`
* Handles class imbalance using `class_weight='balanced'`
* Evaluated using weighted F1-score

---

### 4. Evaluation

* Classification report (precision, recall, F1)
* Confusion matrix visualization
* Feature importance analysis

---

### 5. Prediction

* Accepts new employee data via CSV
* Outputs fatigue risk level per employee

---

## 📊 Example Output

```
employee_id   predicted_fatigue_level
101           High
102           Medium
103           Low
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Joblib

---

## Future Improvements

* Real-world dataset can be used instead of synthetic data
* Addition of REST API (FastAPI/Flask)
* Deployment of the model (Docker / Cloud)

---

## 💡 Why This Project?

Fatigue directly impacts productivity, safety, and decision-making in workplaces.
This project demonstrates how machine learning can be used to proactively identify high-risk employees and improve workplace well-being.

---

## 👤 Author

Deependra Sisodia, MCA
---
