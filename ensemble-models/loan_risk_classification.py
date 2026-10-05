import pandas as pd
from sklearn.svm import SVC


data = {
    'Debt_To_Income': [0.15, 0.20, 0.25, 0.55, 0.60, 0.65, 0.18, 0.50],
    'Late_Payments':  [   0,    1,    0,    3,    4,    5,    1,    3],
    'Default_Risk':   [   0,    0,    0,    1,    1,    1,    0,    1]
}

df = pd.DataFrame(data)

X = df[['Debt_To_Income', 'Late_Payments']]
y = df['Default_Risk']

model = SVC(kernel='rbf')
model.fit(X, y)

new_customer = pd.DataFrame([[0.45, 3]], columns=['Debt_To_Income', 'Late_Payments'])
predictions = model.predict(new_customer)[0]

print("Debt to income: 0.45, Late payments: 3")
print(f"Prediction: {'HIGH RISK' if predictions == 1 else 'LOW RISK'}")