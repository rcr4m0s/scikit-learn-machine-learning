import pandas as pd
from sklearn.cluster import KMeans

# 1. Dataset
df = pd.DataFrame({
    'Income_k': [15, 16, 17, 80, 85, 88, 120, 130, 125],
    'Spending_Score': [39, 81, 6, 77, 89, 90, 20, 15, 18]
})

# 2. Subukan ang iba't ibang bilang ng clusters (k = 1 hanggang 6)
inertia_list = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[['Income_k', 'Spending_Score']])
    inertia_list.append(kmeans.inertia_)

# 3. Print Inertia Values
for k, inertia in zip(range(1, 7), inertia_list):
    print(f"k = {k} | Inertia: {inertia:.2f}")