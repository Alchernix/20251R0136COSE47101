def get_association_rules(freq_itemsets, min_confidence=0.5):
    from mlxtend.frequent_patterns import association_rules

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="confidence", ascending=False)

    return rules