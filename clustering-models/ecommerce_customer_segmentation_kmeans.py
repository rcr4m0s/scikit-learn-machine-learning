import pandas as pd
from sklearn.cluster import KMeans

# 1. Dataset: Purchase Frequency vs Total Spent ($)
data = {
    'Purchases_Per_Year': [2, 3, 1, 25, 30, 28, 12, 15, 10],
    'Total_Spent':        [50, 80, 40, 1200, 1500, 1350, 450, 500, 400]
}
df = pd.DataFrame(data)

# 2. Check Elbow Method (k = 1 to 5)
print("=== INERTIA PER K ===")
for k in range(1, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df[['Purchases_Per_Year', 'Total_Spent']])
    print(f"k = {k} | Inertia: {km.inertia_:,.2f}")

# 3. Final Model (k = 3)
final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Segment'] = final_kmeans.fit_predict(df[['Purchases_Per_Year', 'Total_Spent']])

print("\n=== FINAL SEGMENTATION ===")
print(df)