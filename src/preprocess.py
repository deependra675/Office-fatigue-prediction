import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    low_th = df['fatigue_risk_score'].quantile(0.33)
    high_th = df['fatigue_risk_score'].quantile(0.66)

    def assign_risk(score):  # Classifying fatigue risk
        if score <= low_th:
            return "Low"
        elif score <= high_th:
            return "Medium"
        else:
            return "High"
    
    df['fatigue_risk_level'] = df['fatigue_risk_score'].apply(assign_risk)

    X = df.drop(columns =[      # separate features and target
        'employee_id',
        'fatigue_score',
        'fatigue_risk_score',
        'fatigue_risk_level'
    ])
    y = df['fatigue_risk_level']

    X = pd.get_dummies(X, columns = ['time_of_day'], drop_first=True) # one hot encode categorial feature

    numeric_cols = [ #Scale numerical features
        'hours_worked_today',
        'continous_work_hours',
        'tasks_completed',
        'break_count',
        'previous_day_hours',
        'sleep_proxy'
    ]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y
def prepare_features(df):
    # Drop unnecessary columns if present
    df = df.copy()

    if 'employee_id' in df.columns:
        df = df.drop(columns=['employee_id'])

    # One-hot encoding
    df = pd.get_dummies(df, columns=['time_of_day'], drop_first=False)

    # Ensure all expected columns exist
    expected_cols = [
        'hours_worked_today',
        'continous_work_hours',
        'tasks_completed',
        'break_count',
        'previous_day_hours',
        'sleep_proxy',
        'time_of_day_Afternoon',
        'time_of_day_Night'
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    # Scale
    numeric_cols = [
        'hours_worked_today',
        'continous_work_hours',
        'tasks_completed',
        'break_count',
        'previous_day_hours',
        'sleep_proxy'
    ]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
