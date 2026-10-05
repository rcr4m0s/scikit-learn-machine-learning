import pandas as pd
from xgboost import XGBClassifier

data = {
    'Age':         [25, 30, 45, 60, 65, 70, 28, 55],
    'RestingBP':   [110, 115, 120, 145, 150, 160, 112, 138],
    'MaxHR':       [180, 175, 165, 120, 110, 105, 185, 130],
    'HeartRisk':   [0,   0,   0,   1,   1,   1,   0,   1]
}

df = pd.DataFrame(data)

X = df[['Age', 'RestingBP', 'MaxHR']]
y = df['HeartRisk']

model = XGBClassifier(n_estimators= 50, learning_rate=0.1, max_depth=3, eval_metric='logloss', random_state=42)
model.fit(X, y)

new_patient = pd.DataFrame([[58, 140, 125]], columns=['Age', 'RestingBP', 'MaxHR'])
predictions = model.predict(new_patient)[0]

print("=== PATIENT HEALTH ASSESSMENT ===")
print("Age: 58, RestingBP: 140, MaxHR: 125")
print(f"Prediction: {'HIGH RISK' if predictions == 1 else 'LOW RISK'}")