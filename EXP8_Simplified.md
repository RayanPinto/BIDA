# EXP8 - LDA Topic Modeling

Source file: `EXP8_Simplified.py`

```python
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Files to process
files = ["The Indian Express/TheIndianExpress.html", "The Economic Times/TheEconomicTimes.html", "TIE/TIE.html"]

def extract_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Extract text from paragraphs
    text = " ".join(p.get_text() for p in soup.find_all('p'))
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Extract texts
texts = [extract_text(f) for f in files if len(extract_text(f)) > 50]

# Create bag-of-words
vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.95)
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()

# Find common terms
freq = X.sum(axis=0).A1
top_idx = freq.argsort()[::-1][:15]
print("Top Common Terms:")
for i, idx in enumerate(top_idx):
    print(f"{vocab[idx]:20s} {int(freq[idx])}")

# LDA Topic Modeling
k = min(3, len(texts))
lda = LatentDirichletAllocation(n_components=k, random_state=0)
lda.fit(X)

print("\nHidden Topics (LDA):")
for i, topic in enumerate(lda.components_):
    top_words = topic.argsort()[::-1][:10]
    print(f"Topic {i+1}: {' | '.join(vocab[top_words])}")
``
