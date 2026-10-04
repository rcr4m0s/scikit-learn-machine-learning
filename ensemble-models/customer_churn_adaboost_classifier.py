import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

data = {
    'Monthly_Fee': [20, 25, 80, 95, 100, 30, 85, 110],
    'Tenure_Months': [1,  2, 36, 48, 60,  6, 12,   3],
    'Churn':        [1,  1,  0,  0,  0,  1,  1,   1]
}

df = pd.DataFrame(data)

X = df[['Monthly_Fee', 'Tenure_Months']]
y = df['Churn']

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50, 
    random_state=42
)

model.fit(X, y)

new_customer = pd.DataFrame([[90, 20]], columns=['Monthly_Fee', 'Tenure_Months'])
pred = model.predict(new_customer)

print("Customer ($90/mo, 2 months tenure)")
print(f"Prediction: {'WILL CHURN' if pred == 1 else 'WILL STAY'}")

