"""Evaluate the One-Line Rule baseline (single-seed or multi-seed 95% CI).

Rule: URLs starting with ``https://www.`` are predicted as Legitimate (0);
all other URLs are predicted as Phishing (1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import etld1_split, random_split
from src.evaluation import (
    CORE_METRICS,
    DEFAULT_SEEDS,
    REALISTIC_POSITIVE_RATES,
    evaluate_model,
    normalize_seeds,
)
from src.features import DATA_PATH

TEST_SIZE = 0.2
RANDOM_STATE = 42
SEEDS = DEFAULT_SEEDS
RULE_PREFIX = "https://www."


class OneLineRuleClassifier:
    """Baseline 1: Flag any URL that does not start with ``https://www.``."""

    def __init__(self, prefix: str = RULE_PREFIX) -> None:
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneLineRuleClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        has_prefix = (
            X["URL"].fillna("").astype(str).str.startswith(self.prefix)
        )
        return (~has_prefix).astype(int).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pos_prob = self.predict(X).astype(float)
        return np.column_stack([1.0 - pos_prob, pos_prob])


def extract_url_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the raw URL column and standardized binary labels (1=Phish, 0=Legit)."""
    missing = {"URL", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["label"].isna().any():
        raise ValueError("The label column contains missing values.")

    X = df[["URL"]].copy()
    y = (df["label"] == 0).astype(int)
    return X, y


def build_one_line_rule_model(
    random_state: int = RANDOM_STATE,
) -> OneLineRuleClassifier:
    """Build the stateless one-line rule classifier."""
    return OneLineRuleClassifier()


def print_rule_assumption_check(df: pd.DataFrame) -> None:
    """Verify the prefix assumption across all legitimate and phishing URLs."""
    _, y_true = extract_url_and_labels(df)
    has_prefix = df["URL"].fillna("").astype(str).str.startswith(RULE_PREFIX)

    legit_mask = y_true == 0
    phish_mask = y_true == 1
    legit_total = int(legit_mask.sum())
    phish_total = int(phish_mask.sum())
    legit_match = int((legit_mask & has_prefix).sum())
    phish_match = int((phish_mask & has_prefix).sum())

    print("\n--- One-Line Rule Assumption Check (Full Dataset) ---")
    print(
        f"Legitimate URLs (0): {legit_match:,} / {legit_total:,} "
        f"({100.0 * legit_match / legit_total:.2f}%) start with {RULE_PREFIX!r}"
    )
    print(
        f"Phishing URLs   (1): {phish_match:,} / {phish_total:,} "
        f"({100.0 * phish_match / phish_total:.2f}%) start with {RULE_PREFIX!r}"
    )


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    seeds = normalize_seeds(SEEDS)
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"2. Evaluating One-Line Rule on Random Split (K={len(seeds)} seed(s))...")
    res_random = evaluate_model(
        df,
        split_name="ONE-LINE RULE - RANDOM SPLIT (https://www.)",
        split_fn=lambda d, s: random_split(
            d, test_size=TEST_SIZE, random_state=s, label_col="label"
        ),
        model_builder=build_one_line_rule_model,
        feature_extractor=extract_url_and_labels,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
    )

    print(f"\n3. Evaluating One-Line Rule on eTLD+1 Split (K={len(seeds)} seed(s))...")
    res_etld1 = evaluate_model(
        df,
        split_name="ONE-LINE RULE - eTLD+1 SPLIT (https://www.)",
        split_fn=lambda d, s: etld1_split(
            d, test_size=TEST_SIZE, random_state=s, url_col="URL", n_candidates=100
        ),
        model_builder=build_one_line_rule_model,
        feature_extractor=extract_url_and_labels,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
        assert_etld1_disjoint=True,
    )

    comparison_df = pd.DataFrame(
        [
            {m: res_random["summary"][m]["formatted"] for m in CORE_METRICS},
            {m: res_etld1["summary"][m]["formatted"] for m in CORE_METRICS},
        ],
        index=["Rule - Random Split", "Rule - eTLD+1 Split"],
    )
    suffix = f"K={len(seeds)} seeds, Mean ± 95% CI" if len(seeds) >= 2 else f"seed={seeds[0]}"
    print(f"\n================ ONE-LINE RULE COMPARISON SUMMARY ({suffix}) ================")
    print(comparison_df.to_string())
    print("=============================================================================")

    rand_1pct = res_random["rate_summary_df"].loc[
        res_random["rate_summary_df"]["Target Pos Rate"] == "1.00%",
        list(CORE_METRICS),
    ].iloc[0]
    etld1_1pct = res_etld1["rate_summary_df"].loc[
        res_etld1["rate_summary_df"]["Target Pos Rate"] == "1.00%",
        list(CORE_METRICS),
    ].iloc[0]
    comparison_1pct_df = pd.DataFrame(
        [rand_1pct.to_dict(), etld1_1pct.to_dict()],
        index=["Rule - Random Split (1% Pos)", "Rule - eTLD+1 Split (1% Pos)"],
    )
    print(f"\n================ RULE 1% POSITIVE-RATE COMPARISON SUMMARY ({suffix}) ================")
    print(comparison_1pct_df.to_string())
    print("=====================================================================================")

    print_rule_assumption_check(df)


if __name__ == "__main__":
    main()
