import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

data = {
    'Amount':    [10, 25, 50, 1200, 2500, 3000, 15, 800],
    'Distance':  [2,  5,  1,  500,  800,  1200, 3,  450],
    'Is_Fraud':  [0,  0,  0,  1,    1,    1,    0,  1]
}
df = pd.DataFrame(data)

X = df[['Amount', 'Distance']]
y = df['Is_Fraud']

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X, y)

new_transaction = pd.DataFrame([[1500, 600]], columns=['Amount', 'Distance'])
predictions = model.predict(new_transaction)

print("Transaction ($1,500, 600 miles away)")
print(f"Status: {'FRAUDULENT TRANSACTION' if predictions == 1 else 'LEGIT TRANSACTION'}")