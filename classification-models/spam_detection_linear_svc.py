from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


texts = [
    "Win a free iPhone now", "Free prize money awaiting", "Call me when you are home", 
    "Let's grab lunch today", "Claim your free reward", "Are we still meeting later"
]
labels = [1, 1, 0, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = SVC(kernel='linear')
model.fit(X, labels)

new_msg = ["Free discount code inside"]
new_X = vectorizer.transform(new_msg)
pred = model.predict(new_X)[0]

print(f"Message: '{new_msg[0]}' -> {'SPAM' if pred == 1 else 'NOT SPAM'}")