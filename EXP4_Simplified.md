# EXP4 - Association Rules Mining (Apriori Algorithm)

Source file: `EXP4_Simplified.py`

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load data
df = pd.read_csv("Market_Basket_Optimisation - Market_Basket_Optimisation.csv")

# Convert to transactions
transactions = []
for i in range(len(df)):
    transactions.append([str(df.values[i,j]) for j in range(df.shape[1]) if str(df.values[i,j]) != 'nan'])

# Encode transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
print("Frequent Itemsets:", frequent_itemsets.shape[0])

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1) & rules['consequents'].apply(lambda x: len(x) >= 1)]
print("Association Rules:", rules.shape[0])

# Show top rules
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Top items
all_items = [item for sublist in transactions for item in sublist]
top_items = pd.Series(all_items).value_counts().head(10)
print("\nTop 10 Items:")
print(top_items)
``
