
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Find the project folder automatically.
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def main():
    # 1. Check whether the dataset file exists.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n"
            "Please check the file name and location."
        )

    # 2. Load the real dataset.
    print("Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    # 3. Check that the required columns exist.
    required_columns = {"URL", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns not found: {missing_columns}"
        )

    # 4. Check that labels are the expected binary values.
    if df["label"].isna().any():
        raise ValueError("The label column contains missing values.")

    actual_labels = set(df["label"].unique())

    if actual_labels != {0, 1}:
        raise ValueError(
            f"Expected labels 0 and 1, but found: {actual_labels}"
        )

    # Official UCI convention:
    # 0 = phishing, 1 = legitimate. (in raw csv file)
    # 1 = Phishing, 0 = legitimate.
    y_true = (df["label"] == 0).astype(int)

    # 5. Apply the proposed one-line rule.
    # Starts with https://www. -> predict legitimate (0).
    # Otherwise               -> predict phishing (1).
    url_has_prefix = (
        df["URL"]
        .fillna("")
        .astype(str)
        .str.startswith("https://www.")
    )

    y_pred = (~url_has_prefix).astype(int)

    # 6. Check the rule's key assumption on legitimate URLs.
    legitimate_mask = y_true == 0
    legitimate_count = int(legitimate_mask.sum())
    legitimate_matching = int(
        (legitimate_mask & url_has_prefix).sum()
    )

    matching_percentage = (
        100 * legitimate_matching / legitimate_count
        if legitimate_count
        else 0
    )

    print("\n--- One-line rule assumption check ---")
    print(f"Legitimate URLs (Negative = 0): {legitimate_count:,}")
    print(
        "Legitimate URLs starting with https://www.: "
        f"{legitimate_matching:,}"
    )
    print(
        "Percentage of legitimate URLs matching the prefix: "
        f"{matching_percentage:.2f}%"
    )

    # 7. Evaluate predictions.
    print("\n--- Confusion matrix ---")
    print("Order: legitimate (0 - Negative), phishing (1 - Positive)")
    print("[ TN  FP ]\n[ FN TP ]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\n--- Classification report ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate (0)", "Phishing (1)"],
            digits = 4,
            zero_division=0,
        )
    )

    print("Smoke test completed successfully!")


if __name__ == "__main__":
    main()
