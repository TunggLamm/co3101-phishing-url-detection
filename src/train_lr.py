"""Train and evaluate Logistic Regression (single-seed or multi-seed 95% CI)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import etld1_split, random_split
from src.evaluation import (
    CORE_METRICS,
    DEFAULT_SEEDS,
    REALISTIC_POSITIVE_RATES,
    evaluate_model,
    evaluate_predictions,
    format_metric,
    normalize_seeds,
)
from src.features import (
    DATA_PATH,
    FEATURE_DESCRIPTIONS,
    SELECTED_LEXICAL_FEATURES,
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
SEEDS = DEFAULT_SEEDS


def build_logistic_regression_pipeline(
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a standard pipeline with StandardScaler and LogisticRegression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def print_feature_weights(models: list[Pipeline]) -> None:
    """Print standardized feature weights (single-seed or Mean ± 95% CI)."""
    coefs = np.vstack([m.named_steps["clf"].coef_[0] for m in models])
    means = coefs.mean(axis=0)
    weight_col = "Weight (Mean ± 95% CI)" if len(models) >= 2 else "Weight (w)"

    weights_df = pd.DataFrame(
        {
            "Description": [
                FEATURE_DESCRIPTIONS[col] for col in SELECTED_LEXICAL_FEATURES
            ],
            weight_col: [
                format_metric(coefs[:, i])
                for i in range(len(SELECTED_LEXICAL_FEATURES))
            ],
            "|Weight|": [round(abs(m), 4) for m in means],
            "Direction": [
                "Phishing (+)" if m > 0 else "Legitimate (-)" for m in means
            ],
        },
        index=SELECTED_LEXICAL_FEATURES,
    ).sort_values(by="|Weight|", ascending=False)

    print(
        f"\n--- Standardized Feature Weights (eTLD+1 Split, K={len(models)} seed(s)) ---"
    )
    print(weights_df.to_string())


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    seeds = normalize_seeds(SEEDS)
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"2. Training Logistic Regression on Random Split (K={len(seeds)} seed(s))...")
    res_random = evaluate_model(
        df,
        split_name="LOGISTIC REGRESSION - RANDOM SPLIT (10 LEXICAL FEATURES)",
        split_fn=lambda d, s: random_split(
            d, test_size=TEST_SIZE, random_state=s, label_col="label"
        ),
        model_builder=build_logistic_regression_pipeline,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
    )

    print(f"\n3. Training Logistic Regression on eTLD+1 Split (K={len(seeds)} seed(s))...")
    res_etld1 = evaluate_model(
        df,
        split_name="LOGISTIC REGRESSION - eTLD+1 SPLIT (10 LEXICAL FEATURES)",
        split_fn=lambda d, s: etld1_split(
            d, test_size=TEST_SIZE, random_state=s, url_col="URL", n_candidates=100
        ),
        model_builder=build_logistic_regression_pipeline,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
        assert_etld1_disjoint=True,
    )

    comparison_df = pd.DataFrame(
        [
            {m: res_random["summary"][m]["formatted"] for m in CORE_METRICS},
            {m: res_etld1["summary"][m]["formatted"] for m in CORE_METRICS},
        ],
        index=["Random Split", "eTLD+1 Split"],
    )
    suffix = f"K={len(seeds)} seeds, Mean ± 95% CI" if len(seeds) >= 2 else f"seed={seeds[0]}"
    print(f"\n================ BASELINE COMPARISON SUMMARY ({suffix}) ================")
    print(comparison_df.to_string())
    print("==========================================================================")

    print_feature_weights(res_etld1["models"])


if __name__ == "__main__":
    main()
