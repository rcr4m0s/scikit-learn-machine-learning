import pandas as pd
from sklearn.cluster import AgglomerativeClustering
data = {
    'Age':            [18, 22, 25, 45, 52, 58],
    'SpendingScore':  [80, 90, 85, 20, 15, 30]
}

df = pd.DataFrame(data)

model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
df['Cluster'] = model.fit_predict(df)

print("=== AGGLOMERATIVE CLUSTERING RESULTS ===")
print(df)
