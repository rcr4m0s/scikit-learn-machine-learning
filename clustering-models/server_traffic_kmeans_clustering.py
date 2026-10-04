import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Requests_Per_Sec': [100, 120, 110, 850, 900, 920],
    'Latency_ms':       [15,  18,  12,  200, 210, 195]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

df['Cluster'] = kmeans.fit_predict(df[['Requests_Per_Sec', 'Latency_ms']])

print("=== CLUSTER ASSIGNEMENT ===")
print(df)

print("\n=== CLUSTER CENTROIDS ===")
print(kmeans.cluster_centers_)