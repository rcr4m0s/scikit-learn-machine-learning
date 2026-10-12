import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = {
    'Experience_Yrs': [1, 3, 5, 10, 2, 8, 4, 6, 9, 1],
    'Projects_Done':  [2, 5, 8, 15, 3, 12, 6, 9, 14, 1],
    'Promoted':       [0, 0, 1,  1, 0,  1, 0, 1,  1, 0]
}

df = pd.DataFrame(data)
X = df[['Experience_Yrs', 'Projects_Done']]
y = df['Promoted']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')


print("=== PIPELINE & CROSS-VALIDATION ===")
print(f"Fold Accuracies: {scores}")
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")

pipeline.fit(X, y)
predictions = pipeline.predict(pd.DataFrame([[6, 10]], columns=X.columns))[0]

print(f"\nPredictions for 6 yrs exp & 10 projects: {'PROMOTED' if predictions == 1 else 'NOT PROMOTED'}")