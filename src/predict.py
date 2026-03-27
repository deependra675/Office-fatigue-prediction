import joblib
import pandas as pd
from pathlib import Path
from preprocess import prepare_features

# Defining Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_fatigue.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
INPUT_CSV = BASE_DIR / "data" / "raw" / "predict_input.csv"

if __name__ == "__main__":
    # Loading model and feature columns
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # Loading new employee data
    new_data = pd.read_csv(INPUT_CSV)

    # Prepareing features
    X = prepare_features(new_data)

    # Alignment of columns with training
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    # Makeing predictions
    predictions = model.predict(X)

    # Adding predictions to dataframe and display results
    new_data['predicted_fatigue_level'] = predictions
    print(new_data[['employee_id', 'predicted_fatigue_level']])
