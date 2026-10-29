"""Run both split methods on the downloaded PhiUSIIL CSV and print a summary."""

from pathlib import Path

import pandas as pd

from data_split import (
    assert_no_etld1_overlap,
    etld1_split,
    extract_etld1,
    random_split,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def print_summary(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print row counts, test fraction, and original label counts."""
    total_rows = len(train_df) + len(test_df)
    test_fraction = len(test_df) / total_rows

    print(f"\n--- {name} ---")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print(f"Test row fraction: {test_fraction:.2%}")
    print("Train label counts (original UCI labels):")
    print(train_df["label"].value_counts().sort_index().to_dict())
    print("Test label counts (original UCI labels):")
    print(test_df["label"].value_counts().sort_index().to_dict())


def main() -> None:
    """Load PhiUSIIL, run both splitting methods, and validate domain split."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Put PhiUSIIL_Phishing_URL_Dataset.csv in data/raw/."
        )

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    required_columns = {"URL", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    train_random, test_random = random_split(
        df,
        test_size=0.2,
        random_state=42,
        label_col="label",
    )
    print_summary("Random split", train_random, test_random)

    train_domain, test_domain = etld1_split(
        df,
        test_size=0.2,
        random_state=42,
        url_col="URL",
        n_candidates=100,
    )
    assert_no_etld1_overlap(train_domain, test_domain, url_col="URL")

    train_domains = {extract_etld1(url) for url in train_domain["URL"]}
    test_domains = {extract_etld1(url) for url in test_domain["URL"]}

    print_summary("eTLD+1 domain-disjoint split", train_domain, test_domain)
    print(f"Unique eTLD+1 groups in train: {len(train_domains):,}")
    print(f"Unique eTLD+1 groups in test:  {len(test_domains):,}")
    print(f"Overlapping eTLD+1 groups: {len(train_domains & test_domains)}")
    print("PASS: no eTLD+1 group appears in both train and test.")


if __name__ == "__main__":
    main()
