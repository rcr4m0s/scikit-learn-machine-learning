import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


data = {
    'RAM_GB':        [8,  16,  8,  32,  16,  64],
    'Storage_GB':    [256, 512, 512, 1024, 1024, 2048],
    'Random_Noise':  [100, 105, 98,  102,  99,   101],
    'Price_USD':     [500, 900, 650, 1800, 1200, 3000]
}

df = pd.DataFrame(data)

X = df [['RAM_GB', 'Storage_GB', 'Random_Noise']]
y = df['Price_USD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_lasso = Lasso(alpha=10.0, random_state=42)
model_lasso.fit(X_scaled, y)

coeff = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model_lasso.coef_
})

print("=== LASSO FEATURE COEFFICIENT ===")
print(coeff)

new_laptop = scaler.transform(pd.DataFrame([[16, 512, 100]], columns=X.columns))
prediction = model_lasso.predict(new_laptop)[0]

print(f"\nPredicted Price for 16 GB RAM: {prediction:.2f}")