import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
pipeline.fit(X, y)

model_filename = 'promotion_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model successfully saved to {model_filename}!")

loaded_model = joblib.load(model_filename)

new_applicant = pd.DataFrame([[7, 11]], columns=X.columns)
prediction = loaded_model.predict(new_applicant)[0]

print("=== LOADED MODEL PREDICTION ===")
print(f"Prediction for 7 yrs exp & 11 projects: {'PROMOTED' if prediction == 1 else 'NOT PROMOTED'}")