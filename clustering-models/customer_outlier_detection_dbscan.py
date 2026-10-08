import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = {
    'Annual_Income':   [15, 16, 17, 18, 55, 60, 65, 70, 120, 125, 25],
    'Spending_Score':  [39, 81, 6,  77, 40, 35, 60, 80, 20,  15,  5]
}
df = pd.DataFrame(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

model = DBSCAN(eps=0.8, min_samples=2)
df['Cluster'] = model.fit_predict(X_scaled)

print("=== DBSCAN CLUSTERING & NOISE DETECTION ===")
print(df)
