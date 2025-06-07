def get_frequent_itemsets(min_sup=0.5):
    from preprocess_data import get_preprocessed_data
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori
    import pandas as pd

    df = get_preprocessed_data("raw_data.xlsx", "course_name.json")

    transactions = df[df.columns[1:]].apply(
            lambda row: sorted(set(sum(row, []))), axis=1
        ).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

    return frequent_itemsets