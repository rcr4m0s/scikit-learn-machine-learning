import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

data = {
    'Income_k':     [30, 45, 60, 80, 25, 90, 35, 70],
    'Credit_Score': [580, 620, 710, 750, 530, 800, 590, 730],
    'Default':      [1, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['Income_k', 'Credit_Score']]
y = df['Default']

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(random_state=42)
clf3 = SVC(probability=True, kernel='rbf')

ensemble_model = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='soft'
)

ensemble_model.fit(X, y)

new_applicant = pd.DataFrame([[50, 680]], columns=X.columns)
prediction = ensemble_model.predict(new_applicant)[0]
proba = ensemble_model.predict_proba(new_applicant)[0]

print("=== VOTING CLASSIFIER ENSEMBLE ===")
print(f"Prediction for $50k Income & 680 Credit Score: {'DEFAULT (HIGH RISK)' if prediction == 1 else 'NO DEFAULT (LOW RISK)'}")
print(f"Probabilities [No Default, Default]: {proba}")