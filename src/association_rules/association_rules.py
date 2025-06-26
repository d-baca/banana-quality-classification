import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def generate_transactions(df: pd.DataFrame) -> list[list[str]]:
    """
    Kolumny ripeness_bin i acidity_bin już istnieją.
    Dodaje dyskretyzację size i sweetness do 3 kwantyli.
    """
    df['size_bin'] = pd.qcut(df['Size'], 3, labels=['low', 'medium', 'high'])
    df['sweetness_bin'] = pd.qcut(df['Sweetness'], 3, labels=[
                                  'low', 'medium', 'high'])

    transactions = []
    for _, row in df.iterrows():
        items = [
            f"ripeness={row['ripeness_bin']}",
            f"acidity={row['acidity_bin']}",
            f"size={row['size_bin']}",
            f"sweetness={row['sweetness_bin']}",
            f"quality={row['Quality']}"
        ]
        transactions.append(items)
    return transactions


def mine_rules(transactions: list[list[str]],
               min_support: float = 0.05,
               min_confidence: float = 0.7) -> pd.DataFrame:
    """Generuje reguły asocjacyjne z transakcji."""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)
    freq_items = apriori(df_tf, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items,
                              metric="confidence",
                              min_threshold=min_confidence)
    return rules


def main(input_csv: str,
         output_dir: str,
         min_support: float,
         min_confidence: float):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    tx = generate_transactions(df)
    rules = mine_rules(tx, min_support, min_confidence)

    # Zostawia tylko reguły, które przewidują quality=Good
    good_rules = rules[
        rules['consequents'].apply(lambda s: 'quality=Good' in s)
    ].copy()
    # sortowanie według confidence i lift, wybranie top5
    top5 = good_rules.sort_values(
        ['confidence', 'lift'], ascending=False
    ).head(5)

    # Zapisanie wszystkich reguł i top5 do plików CSV
    rules.to_csv(os.path.join(output_dir, "all_rules.csv"), index=False)
    top5.to_csv(os.path.join(output_dir, "top5_good_rules.csv"), index=False)

    print("\nTop-5 rules predicting Good:\n")
    print(top5[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Association rules on banana data")
    parser.add_argument("--input",       required=True,
                        help="CSV z banana_preprocessed.csv")
    parser.add_argument("--output-dir",  default="assoc_rules",
                        help="gdzie zapisać reguły")
    parser.add_argument("--min-support",  type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()
    main(args.input, args.output_dir,
         args.min_support, args.min_confidence)
