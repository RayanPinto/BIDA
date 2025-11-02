"""
Script to create a comprehensive text file with all simplified programs
Uses only built-in Python libraries - can be easily converted to PDF
"""

import os
from datetime import datetime

def create_comprehensive_document():
    """Create a comprehensive text document with all programs"""
    
    output_file = "Simplified_Lab_Programs_Complete.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("SIMPLIFIED LAB PROGRAMS FOR EXAM\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write("This document contains simplified versions of all lab programs that use only\n")
        f.write("built-in Python libraries. These programs are designed to be easy to understand\n")
        f.write("and implement during exams without internet access.\n\n")
        
        f.write("PROGRAMS INCLUDED:\n")
        f.write("1. EXP4 - Association Rules Mining (Apriori Algorithm)\n")
        f.write("2. EXP5 - Sentiment Analysis using Naive Bayes\n")
        f.write("3. EXP6 - Web Scraping and Topic Modeling\n")
        f.write("4. EXP7 - HTML Sentiment Analysis\n")
        f.write("5. EXP8 - LDA Topic Modeling\n\n")
        
        f.write("COMMON FEATURES:\n")
        f.write("- No external dependencies\n")
        f.write("- Uses only Python standard library\n")
        f.write("- Clear comments and documentation\n")
        f.write("- Error handling included\n")
        f.write("- Easy to understand and modify\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PROGRAM 1: EXP4 - ASSOCIATION RULES MINING\n")
        f.write("=" * 80 + "\n\n")
        
        # Add EXP4 program
        if os.path.exists("EXP4_Simplified.py"):
            f.write("File: EXP4_Simplified.py\n")
            f.write("Purpose: Find frequent itemsets and association rules from market basket data\n")
            f.write("Data: Market_Basket_Optimisation - Market_Basket_Optimisation.csv\n\n")
            f.write("CODE:\n")
            f.write("-" * 40 + "\n")
            with open("EXP4_Simplified.py", 'r', encoding='utf-8') as exp4_file:
                f.write(exp4_file.read())
            f.write("\n\n")
        else:
            f.write("EXP4_Simplified.py not found!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PROGRAM 2: EXP5 - SENTIMENT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Add EXP5 program
        if os.path.exists("EXP5_Simplified.py"):
            f.write("File: EXP5_Simplified.py\n")
            f.write("Purpose: Analyze sentiment of customer reviews using Naive Bayes classifier\n")
            f.write("Data: customer_review - customer_review.csv\n\n")
            f.write("CODE:\n")
            f.write("-" * 40 + "\n")
            with open("EXP5_Simplified.py", 'r', encoding='utf-8') as exp5_file:
                f.write(exp5_file.read())
            f.write("\n\n")
        else:
            f.write("EXP5_Simplified.py not found!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PROGRAM 3: EXP6 - WEB SCRAPING AND TOPIC MODELING\n")
        f.write("=" * 80 + "\n\n")
        
        # Add EXP6 program
        if os.path.exists("EXP6_Simplified.py"):
            f.write("File: EXP6_Simplified.py\n")
            f.write("Purpose: Extract text from HTML files and perform topic modeling\n")
            f.write("Data: HTML files in Artificial intelligence/ and Startups_TechCrunch/ folders\n\n")
            f.write("CODE:\n")
            f.write("-" * 40 + "\n")
            with open("EXP6_Simplified.py", 'r', encoding='utf-8') as exp6_file:
                f.write(exp6_file.read())
            f.write("\n\n")
        else:
            f.write("EXP6_Simplified.py not found!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PROGRAM 4: EXP7 - HTML SENTIMENT ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Add EXP7 program
        if os.path.exists("EXP7_Simplified.py"):
            f.write("File: EXP7_Simplified.py\n")
            f.write("Purpose: Extract comments from HTML files and analyze their sentiment\n")
            f.write("Data: r_news/r_news.html\n\n")
            f.write("CODE:\n")
            f.write("-" * 40 + "\n")
            with open("EXP7_Simplified.py", 'r', encoding='utf-8') as exp7_file:
                f.write(exp7_file.read())
            f.write("\n\n")
        else:
            f.write("EXP7_Simplified.py not found!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PROGRAM 5: EXP8 - LDA TOPIC MODELING\n")
        f.write("=" * 80 + "\n\n")
        
        # Add EXP8 program
        if os.path.exists("EXP8_Simplified.py"):
            f.write("File: EXP8_Simplified.py\n")
            f.write("Purpose: Perform topic modeling on news articles using simplified LDA\n")
            f.write("Data: HTML files in The Indian Express/, The Economic Times/, and TIE/ folders\n\n")
            f.write("CODE:\n")
            f.write("-" * 40 + "\n")
            with open("EXP8_Simplified.py", 'r', encoding='utf-8') as exp8_file:
                f.write(exp8_file.read())
            f.write("\n\n")
        else:
            f.write("EXP8_Simplified.py not found!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("USAGE INSTRUCTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. PREPARATION:\n")
        f.write("   - Copy all Python files to your exam directory\n")
        f.write("   - Ensure CSV and HTML files are in the correct locations\n")
        f.write("   - Test each program before the exam\n\n")
        
        f.write("2. RUNNING PROGRAMS:\n")
        f.write("   - Each program can be run independently\n")
        f.write("   - No installation of external libraries required\n")
        f.write("   - Programs will automatically find and process the data files\n\n")
        
        f.write("3. UNDERSTANDING THE CODE:\n")
        f.write("   - Each program is well-commented\n")
        f.write("   - Variable names are descriptive\n")
        f.write("   - Functions are small and focused\n")
        f.write("   - Logic is straightforward and easy to follow\n\n")
        
        f.write("4. MODIFYING PARAMETERS:\n")
        f.write("   - Support and confidence thresholds can be adjusted\n")
        f.write("   - Number of topics can be changed\n")
        f.write("   - Minimum word frequencies can be modified\n")
        f.write("   - All parameters are clearly marked in the code\n\n")
        
        f.write("5. TROUBLESHOOTING:\n")
        f.write("   - File not found errors: Check file paths and ensure files exist\n")
        f.write("   - Empty results: Try lowering thresholds or check data quality\n")
        f.write("   - Memory issues: Reduce number of iterations or vocabulary size\n")
        f.write("   - Encoding errors: Ensure files are saved with UTF-8 encoding\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("KEY ALGORITHMS IMPLEMENTED\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Apriori Algorithm:\n")
        f.write("   - Generates candidate itemsets\n")
        f.write("   - Prunes based on support threshold\n")
        f.write("   - Calculates association rules\n\n")
        
        f.write("2. Naive Bayes:\n")
        f.write("   - Calculates prior probabilities\n")
        f.write("   - Computes likelihood with smoothing\n")
        f.write("   - Makes predictions using log probabilities\n\n")
        
        f.write("3. TF-IDF:\n")
        f.write("   - Calculates term frequencies\n")
        f.write("   - Computes inverse document frequencies\n")
        f.write("   - Generates TF-IDF scores\n\n")
        
        f.write("4. LDA (Latent Dirichlet Allocation):\n")
        f.write("   - Implements Gibbs sampling\n")
        f.write("   - Updates topic assignments\n")
        f.write("   - Calculates topic-word distributions\n\n")
        
        f.write("5. Sentiment Analysis:\n")
        f.write("   - Lexicon-based approach\n")
        f.write("   - Handles negation and intensifiers\n")
        f.write("   - Calculates sentiment scores\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("These simplified programs provide all the functionality of the original complex\n")
        f.write("programs while being easy to understand and implement. They use only built-in\n")
        f.write("Python libraries, making them perfect for exam environments without internet access.\n")
        f.write("Each program is self-contained and can be run independently to produce meaningful results.\n\n")
        
        f.write("Good luck with your exam!\n")
        f.write("=" * 80 + "\n")
    
    print(f"Comprehensive document created: {output_file}")
    print("You can now print this file or convert it to PDF using any text-to-PDF converter.")

if __name__ == "__main__":
    create_comprehensive_document()
