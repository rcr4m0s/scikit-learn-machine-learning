import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

data = {
    'Age':          [22, 25, 47, 52, 46, 56, 30, 24, 60, 40],
    'Tenure_Months': [2,  6,  24, 36, 12, 48,  3,  1,  60, 18],
    'Monthly_Charges': [50, 55, 90, 100, 85, 110, 60, 45, 120, 80],
    'Churn':        [1,  1,  0,  0,  0,  0,   1,  1,  0,  1]
}
df = pd.DataFrame(data)

X = df[['Age', 'Tenure_Months', 'Monthly_Charges']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, None],
    'min_samples_split': [2, 4]
}

grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("=== GRID SEARCH BEST PARAMETERS ===")
print(grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_ * 100:.2f}%")

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")