import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance':    [50, 60, 55, 70, 75, 80, 85, 90, 95, 98],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance']]
y = df['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, predictions))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions))