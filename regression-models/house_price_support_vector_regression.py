import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


df = pd.DataFrame({
    'House_Age': [1, 3, 5, 10, 15, 20],
    'Square_Feet': [800, 1000, 1200, 1500, 1800, 2000],
    'Price': [150000, 180000, 210000, 250000, 280000, 310000]
})

X = df[['House_Age', 'Square_Feet']]
y = df['Price']


scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

model = SVR(kernel='linear', C=1000)
model.fit(X_scaled, y)

new_house = scaler_X.transform(pd.DataFrame([[7, 1300]], columns=['House_Age', 'Square_Feet']))
predicted_price = model.predict(new_house)[0]

print(f"Predicted Price for House (Age: 7, SqFt: 1300): ${predicted_price:.2f}")