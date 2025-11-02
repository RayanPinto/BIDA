# Simplified Lab Programs Documentation

## Overview
This document contains simplified versions of all lab programs that use only built-in Python libraries. These programs are designed to be easy to understand and implement during exams without internet access.

## Programs Included

### 1. EXP4 - Association Rules Mining (Apriori Algorithm)
**File:** `EXP4_Simplified.py`
**Purpose:** Find frequent itemsets and association rules from market basket data
**Data:** `Market_Basket_Optimisation - Market_Basket_Optimisation.csv`

**Key Features:**
- Implements Apriori algorithm from scratch
- Finds frequent itemsets of different sizes
- Generates association rules with support, confidence, and lift
- Shows top purchased items
- Uses only `csv`, `collections`, `itertools`, and `math` libraries

**How to Run:**
```bash
python EXP4_Simplified.py
```

**Expected Output:**
- Top 10 most purchased items
- Total frequent itemsets found
- Association rules with metrics
- Rules sorted by confidence

### 2. EXP5 - Sentiment Analysis using Naive Bayes
**File:** `EXP5_Simplified.py`
**Purpose:** Analyze sentiment of customer reviews using Naive Bayes classifier
**Data:** `customer_review - customer_review.csv`

**Key Features:**
- Custom Naive Bayes implementation
- Text preprocessing and tokenization
- Train-test split functionality
- Confusion matrix display
- Interactive testing mode
- Uses only built-in libraries

**How to Run:**
```bash
python EXP5_Simplified.py
```

**Expected Output:**
- Class distribution
- Training and test sample counts
- Accuracy score
- Confusion matrix
- Sample predictions
- Interactive testing interface

### 3. EXP6 - Web Scraping and Topic Modeling
**File:** `EXP6_Simplified.py`
**Purpose:** Extract text from HTML files and perform topic modeling
**Data:** HTML files in `Artificial intelligence/` and `Startups_TechCrunch/` folders

**Key Features:**
- Custom HTML parser
- Text extraction and cleaning
- TF-IDF calculation
- Simple topic modeling using word co-occurrence
- Stopword removal
- Uses only built-in libraries

**How to Run:**
```bash
python EXP6_Simplified.py
```

**Expected Output:**
- Processing status for each HTML file
- Top TF-IDF terms (trends)
- Simple topic modeling results
- Document statistics
- Most common words overall

### 4. EXP7 - HTML Sentiment Analysis
**File:** `EXP7_Simplified.py`
**Purpose:** Extract comments from HTML files and analyze their sentiment
**Data:** `r_news/r_news.html`

**Key Features:**
- Comment extraction from HTML
- Lexicon-based sentiment analysis
- Negation handling
- Intensifier detection
- Sentiment distribution analysis
- Interactive testing
- Uses only built-in libraries

**How to Run:**
```bash
python EXP7_Simplified.py
```

**Expected Output:**
- Number of comments found
- Overall sentiment tone
- Sentiment distribution
- Sample positive and negative comments
- Interactive testing interface

### 5. EXP8 - LDA Topic Modeling
**File:** `EXP8_Simplified.py`
**Purpose:** Perform topic modeling on news articles using simplified LDA
**Data:** HTML files in `The Indian Express/`, `The Economic Times/`, and `TIE/` folders

**Key Features:**
- Custom LDA implementation using Gibbs sampling
- Document-term matrix creation
- Topic word extraction
- Vocabulary filtering
- Stopword removal
- Uses only built-in libraries

**How to Run:**
```bash
python EXP8_Simplified.py
```

**Expected Output:**
- Processing status for each HTML file
- Vocabulary size
- Most common words
- Document-term matrix shape
- LDA fitting progress
- Hidden topics with word probabilities

## Common Features Across All Programs

### 1. No External Dependencies
All programs use only Python standard library modules:
- `csv` - for reading CSV files
- `collections` - for Counter, defaultdict
- `itertools` - for combinations
- `math` - for mathematical operations
- `random` - for random sampling
- `re` - for regular expressions
- `os` - for file operations
- `html.parser` - for HTML parsing

### 2. Error Handling
- File not found errors
- Empty data handling
- Invalid input validation
- Graceful error messages

### 3. User-Friendly Output
- Clear progress indicators
- Formatted tables and results
- Interactive testing where applicable
- Detailed statistics and metrics

### 4. Modular Design
- Separate classes for different functionalities
- Reusable functions
- Clear separation of concerns
- Easy to understand and modify

## How to Use During Exam

### 1. Preparation
- Copy all Python files to your exam directory
- Ensure CSV and HTML files are in the correct locations
- Test each program before the exam

### 2. Running Programs
- Each program can be run independently
- No installation of external libraries required
- Programs will automatically find and process the data files

### 3. Understanding the Code
- Each program is well-commented
- Variable names are descriptive
- Functions are small and focused
- Logic is straightforward and easy to follow

### 4. Modifying Parameters
- Support and confidence thresholds can be adjusted
- Number of topics can be changed
- Minimum word frequencies can be modified
- All parameters are clearly marked in the code

## Troubleshooting

### Common Issues:
1. **File not found errors**: Check file paths and ensure files exist
2. **Empty results**: Try lowering thresholds or check data quality
3. **Memory issues**: Reduce number of iterations or vocabulary size
4. **Encoding errors**: Ensure files are saved with UTF-8 encoding

### Solutions:
1. Verify file paths in the code
2. Check data file formats and content
3. Adjust parameters for your specific dataset
4. Use smaller datasets for testing

## Key Algorithms Implemented

### 1. Apriori Algorithm
- Generates candidate itemsets
- Prunes based on support threshold
- Calculates association rules

### 2. Naive Bayes
- Calculates prior probabilities
- Computes likelihood with smoothing
- Makes predictions using log probabilities

### 3. TF-IDF
- Calculates term frequencies
- Computes inverse document frequencies
- Generates TF-IDF scores

### 4. LDA (Latent Dirichlet Allocation)
- Implements Gibbs sampling
- Updates topic assignments
- Calculates topic-word distributions

### 5. Sentiment Analysis
- Lexicon-based approach
- Handles negation and intensifiers
- Calculates sentiment scores

## Performance Notes

- Programs are optimized for clarity over speed
- Some operations may be slower than optimized libraries
- Suitable for small to medium datasets
- Memory usage is kept minimal

## Conclusion

These simplified programs provide all the functionality of the original complex programs while being easy to understand and implement. They use only built-in Python libraries, making them perfect for exam environments without internet access. Each program is self-contained and can be run independently to produce meaningful results.
