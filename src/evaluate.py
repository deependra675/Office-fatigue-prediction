import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from preprocess import load_and_prepare_data

# Defining paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fatigue_10k.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_fatigue.pkl"

# Loading trained model
model = joblib.load(MODEL_PATH)

# Loading and preprocessing dataset
X, y = load_and_prepare_data(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# generating predictions
y_pred = model.predict(X_test)

# Printing detailed classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Plotting Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fatigue Risk Confusion Matrix")
plt.show()

# Plotting feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10,4))
plt.title("Random Forest Feature Importance")
plt.show()
