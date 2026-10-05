import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

texts = [
    "Win free money now", 
    "Claim your urgent prize", 
    "Hey, are we still eating lunch?", 
    "Meeting rescheduled to 3pm"
]
labels = [1, 1, 0, 0]  # 1 = Spam, 0 = Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

model = MultinomialNB()
model.fit(X, labels)

test_msg = ["Free prize for you"]
X_test = vectorizer.transform(test_msg).toarray()
pred = model.predict(X_test)[0]

print(f"Message: '{test_msg[0]}'")
print(f"Result: {'SPAM' if pred == 1 else 'NOT SPAM'}")