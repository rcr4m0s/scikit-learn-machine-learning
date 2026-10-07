import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



data = {
    'Annual_Spend_k':     [10, 15, 20, 50, 60, 55, 120, 140, 110],
    'Purchase_Frequency': [5,  8,  6,  20, 25, 22, 50,  60,  55],
    'Account_Age_Years':  [1,  1,  2,  3,  4,  3,  6,   8,   7],
    'Customer_Tier':      [0,  0,  0,  1,  1,  1,  2,   2,   2]  # 0=Standard, 1=Premium, 2=VIP
}

df = pd.DataFrame(data)

X = df[['Annual_Spend_k', 'Purchase_Frequency', 'Account_Age_Years']]
y = df['Customer_Tier']

model = LinearDiscriminantAnalysis(n_components=2)
model.fit(X, y)

new_customer = pd.DataFrame([[85, 42, 5]], columns=X.columns)
predictions = model.predict(new_customer)[0]
tier = {0: 'Standard', 1: 'Premium', 2: 'VIP'}

print("=== CUSTOMER TIER PREDICTION ===")
print("Annual Spend: $85k | Freq: 42 | Age: 5 yrs")
print(f"Predictions: {tier[predictions]}")