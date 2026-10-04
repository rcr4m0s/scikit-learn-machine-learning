import pandas as pd
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'Packet_Size_KB': [50, 60, 55, 500, 520, 510, 1200, 1250, 1220],
    'Requests_Per_Min': [10, 12, 15, 300, 310, 305, 50, 45, 60]
})

inertia_list = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[['Packet_Size_KB', 'Requests_Per_Min']])
    inertia_list.append(kmeans.inertia_)

for k, inertia in zip(range(1, 7), inertia_list):
    print(f"k = {k} | Inertia: {inertia:,.2f}")