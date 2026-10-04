import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.DataFrame({
    'Age': [18, 20, 22, 45, 50, 52],
    'Annual_Income': [15000, 18000, 12000, 85000, 90000, 95000]
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Annual_Income']])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("=== CLUSTER ASSIGNMENTS WITH SCALING ===")
print(df)

print("\n=== SCALED CENTROIDS ===")
print(kmeans.cluster_centers_)