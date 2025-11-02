# EXP5 - Sentiment Analysis using Naive Bayes

Source file: `EXP5_Simplified.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("customer_review - customer_review.csv")

# Auto-detect columns
text_col = None
label_col = None
for col in df.columns:
    if any(x in col.lower() for x in ['review', 'text', 'comment']):
        text_col = col
    if any(x in col.lower() for x in ['sentiment', 'label', 'rating']):
        label_col = col

# Prepare data
X = df[text_col].astype(str)
y = df[label_col].apply(lambda x: "Positive" if x > 2 else ("Negative" if x <= 2 else "Neutral"))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Predict
y_pred = clf.predict(X_test_vec)

# Results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
``
