import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = {
    'Amount_USD':     [10, 15, 20, 35, 12, 18, 25, 30, 22, 5000],
    'Foreign_IP':     [0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
    'Is_Fraud':       [0,  0,  0,  0,  0,  0,  0,  0,  0,  1] 
}
df = pd.DataFrame(data)

X = df[['Amount_USD', 'Foreign_IP']]
y = df['Is_Fraud']

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

df['Predicted'] = model.predict(X)

print("=== IMBALANCED DATASET PREDICTIONS ===")
print(df[['Amount_USD', 'Is_Fraud', 'Predicted']])

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y, df['Predicted'], zero_division=0))