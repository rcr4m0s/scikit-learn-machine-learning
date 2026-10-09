import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    'Review': [
        "Amazing product, super fast and reliable", 
        "Terrible experience, broke down instantly", 
        "Very happy with this purchase, highly recommended", 
        "Waste of money, poor quality", 
        "Absolute garbage, do not buy", 
        "Exceeded expectations, will buy again"
    ],
    'Sentiment': [1, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

model = MultinomialNB()
model.fit(X_vectorized, y)

new_review = ["Very fast and excellent quality"]
new_vectorized = vectorizer.transform(new_review)
prediction = model.predict(new_vectorized)[0]

print("=== TEXT SENTIMENT PREDICTION ===")
print(f"Review: '{new_review[0]}'")
print(f"Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")