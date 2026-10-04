import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.DataFrame({
    'Feature_1': [0, 1, -1, 0, 5, -5, 6, -6],
    'Feature_2': [1, 0, 0, -1, 5, -5, -6, 6],
    'Class': [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[['Feature_1', 'Feature_2']]
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', C=1.0)
model.fit(X_scaled, y)

test_pt = scaler.transform(pd.DataFrame([[0.5, 0.5]], columns=['Feature_1', 'Feature_2']))

print(f"RBF Prediction for [0.5, 0.5]: Class {model.predict(test_pt)[0]}")