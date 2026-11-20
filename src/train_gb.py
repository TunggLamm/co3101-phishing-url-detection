"""Train and evaluate Gradient Boosting (single-seed or multi-seed 95% CI)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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


def build_gb_model(
    random_state: int = RANDOM_STATE,
) -> GradientBoostingClassifier:
    """Build the Gradient Boosting classifier (no StandardScaler required)."""
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state,
    )


def print_feature_importances(models: list[GradientBoostingClassifier]) -> None:
    """Print tree-based feature importances (single-seed or Mean ± 95% CI)."""
    imps = np.vstack([m.feature_importances_ for m in models])
    means = imps.mean(axis=0)
    imp_col = "Importance (Mean ± 95% CI)" if len(models) >= 2 else "Importance"

    importance_df = pd.DataFrame(
        {
            "Description": [
                FEATURE_DESCRIPTIONS[col] for col in SELECTED_LEXICAL_FEATURES
            ],
            imp_col: [
                format_metric(imps[:, i])
                for i in range(len(SELECTED_LEXICAL_FEATURES))
            ],
            "Share (%)": [f"{m * 100:.2f}%" for m in means],
            "_mean": means,
        },
        index=SELECTED_LEXICAL_FEATURES,
    ).sort_values(by="_mean", ascending=False).drop(columns=["_mean"])

    print(
        f"\n--- Gradient Boosting Feature Importances "
        f"(eTLD+1 Split, K={len(models)} seed(s)) ---"
    )
    print(importance_df.to_string())


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    seeds = normalize_seeds(SEEDS)
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"2. Training Gradient Boosting on Random Split (K={len(seeds)} seed(s))...")
    res_random = evaluate_model(
        df,
        split_name="GRADIENT BOOSTING - RANDOM SPLIT (10 LEXICAL FEATURES)",
        split_fn=lambda d, s: random_split(
            d, test_size=TEST_SIZE, random_state=s, label_col="label"
        ),
        model_builder=build_gb_model,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
    )

    print(f"\n3. Training Gradient Boosting on eTLD+1 Split (K={len(seeds)} seed(s))...")
    res_etld1 = evaluate_model(
        df,
        split_name="GRADIENT BOOSTING - eTLD+1 SPLIT (10 LEXICAL FEATURES)",
        split_fn=lambda d, s: etld1_split(
            d, test_size=TEST_SIZE, random_state=s, url_col="URL", n_candidates=100
        ),
        model_builder=build_gb_model,
        seeds=seeds,
        target_rates=REALISTIC_POSITIVE_RATES,
        assert_etld1_disjoint=True,
    )

    comparison_df = pd.DataFrame(
        [
            {m: res_random["summary"][m]["formatted"] for m in CORE_METRICS},
            {m: res_etld1["summary"][m]["formatted"] for m in CORE_METRICS},
        ],
        index=["GB - Random Split", "GB - eTLD+1 Split"],
    )
    suffix = f"K={len(seeds)} seeds, Mean ± 95% CI" if len(seeds) >= 2 else f"seed={seeds[0]}"
    print(f"\n================ GRADIENT BOOSTING COMPARISON SUMMARY ({suffix}) ================")
    print(comparison_df.to_string())
    print("================================================================================")

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
        index=["GB - Random Split (1% Pos)", "GB - eTLD+1 Split (1% Pos)"],
    )
    print(f"\n================ GB 1% POSITIVE-RATE COMPARISON SUMMARY ({suffix}) ================")
    print(comparison_1pct_df.to_string())
    print("===================================================================================")

    print_feature_importances(res_etld1["models"])


if __name__ == "__main__":
    main()
