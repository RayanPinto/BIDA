from bs4 import BeautifulSoup
import re
from collections import Counter

# Sentiment lexicon
positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'like', 'nice', 'awesome', 'helpful', 'fantastic', 'wonderful', 'brilliant', 'outstanding', 'perfect', 'best', 'better', 'super', 'cool'}
negative_words = {'bad', 'terrible', 'awful', 'hate', 'slow', 'buggy', 'confusing', 'broken', 'issue', 'problem', 'worst', 'disappointed', 'frustrating', 'annoying', 'horrible', 'disgusting', 'pathetic', 'useless', 'worthless', 'stupid', 'dumb', 'idiotic', 'ridiculous', 'absurd', 'nonsense', 'garbage', 'trash'}
negation_words = {'not', 'no', 'never', 'none', 'hardly', 'barely', 'scarcely'}

def extract_comments(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Remove scripts and styles
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    
    # Find comment elements
    comments = []
    for elem in soup.find_all(['div', 'p', 'span', 'article']):
        if any(word in elem.get('class', []) for word in ['comment', 'reply', 'review']):
            text = elem.get_text().strip()
            if len(text) > 12:
                comments.append(text)
    
    return comments

def analyze_sentiment(text):
    words = re.findall(r'\b\w+\b', text.lower())
    score = 0
    for i, word in enumerate(words):
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
        
        # Check for negation
        if i > 0 and words[i-1] in negation_words:
            score *= -1
    
    return "positive" if score > 0 else ("negative" if score < 0 else "neutral")

# Process file
comments = extract_comments("r_news/r_news.html")
if not comments:
    print("No comments found")
else:
    sentiments = [analyze_sentiment(comment) for comment in comments]
    counts = Counter(sentiments)
    
    print(f"Overall tone: {max(counts, key=counts.get)}")
    print(f"Counts: {dict(counts)}")
    
    # Show examples
    print("\nSample comments:")
    for i, comment in enumerate(comments[:5]):
        print(f"{i+1}. [{sentiments[i]}] {comment[:100]}...")