# EXP6 - Web Scraping and Topic Modeling

Source file: `EXP6_Simplified.py`

```python
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# Files to process
files = ["Artificial intelligence/Artificialintelligence.html", "Startups_TechCrunch/Startups_TechCrunch.html"]

def extract_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text()

# Extract text
docs = [extract_text(f) for f in files if len(extract_text(f)) > 200]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.9, min_df=1)
X = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names_out()
scores = X.mean(axis=0).A1

# Top terms
top_idx = scores.argsort()[::-1][:25]
print("Top TF-IDF terms:")
for i, idx in enumerate(top_idx):
    print(f"{terms[idx]:30s} {scores[idx]:.4f}")

# Topic modeling
k = min(6, len(docs))
nmf = NMF(n_components=k, random_state=42)
W = nmf.fit_transform(X)
H = nmf.components_

print("\nTopics:")
for i, topic in enumerate(H):
    top_words = topic.argsort()[::-1][:10]
    print(f"Topic {i}: {', '.join(terms[top_words])}")
``
