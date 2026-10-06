import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    'Bedrooms':   [2, 3, 3, 4, 4, 5, 2, 3],
    'SqFt':       [800, 1200, 1500, 2200, 2500, 3100, 950, 1400],
    'House_Age':  [30, 20, 15, 5, 2, 1, 25, 10],
    'Is_HighPrice': [0, 0, 0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[['Bedrooms', 'SqFt', 'House_Age']]
y = df['Is_HighPrice']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("=== FEATURE IMPORTANCE RANKING ===")
print(importance_df)