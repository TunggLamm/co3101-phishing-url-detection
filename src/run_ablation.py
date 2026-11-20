import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import assert_no_etld1_overlap, etld1_split
from src.evaluation import (
    CORE_METRICS,
    DEFAULT_SEEDS,
    evaluate_predictions,
    format_metric,
    normalize_seeds,
)
from src.features import (
    DATA_PATH,
    NUMERIC_LEAKAGE_COLS,
    extract_features_and_labels,
    get_ablation_feature_sets,
)
from src.train_gb import build_gb_model
from src.train_lr import build_logistic_regression_pipeline
from src.utils import adjust_positive_rate

TEST_SIZE = 0.2
RANDOM_STATE = 42
SEEDS = RANDOM_STATE
REALISTIC_POSITIVE_RATES = 0.01


def evaluate_at_rate(
    y_test: pd.Series,
    y_pred,
    y_prob,
    target_rate: float | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    if target_rate is None:
        return evaluate_predictions(y_test, y_pred, y_prob)
    eval_df = pd.DataFrame({
        "label": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    adj_df = adjust_positive_rate(
        eval_df,
        target_rate=target_rate,
        label_col="label",
        pos_label=1,
        random_state=random_state,
    )
    return evaluate_predictions(adj_df["label"], adj_df["y_pred"], adj_df["y_prob"])


def run_feature_ablation(
    model_name: str,
    model_builder,
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    feature_set = get_ablation_feature_sets(splits[0][1])
    rows = []
    base_f1_orig = None
    base_f1_1pct = None

    for stage_name, cols in feature_set.items():
        orig_list = []
        pct1_list = []

        for seed, train_df, test_df in splits:
            X_train, y_train = extract_features_and_labels(train_df, cols)
            X_test, y_test = extract_features_and_labels(test_df, cols)
            model = model_builder(seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            orig_list.append(evaluate_at_rate(y_test, y_pred, y_prob, target_rate=None))
            pct1_list.append(
                evaluate_at_rate(
                    y_test,
                    y_pred,
                    y_prob,
                    target_rate=REALISTIC_POSITIVE_RATES,
                    random_state=seed,
                )
            )

        f1_orig = [m["F1-score"] for m in orig_list]
        f1_1pct = [m["F1-score"] for m in pct1_list]

        if base_f1_orig is None:
            base_f1_orig = f1_orig
            base_f1_1pct = f1_1pct

        drops_orig = [cur - base for cur, base in zip(f1_orig, base_f1_orig)]
        drops_1pct = [cur - base for cur, base in zip(f1_1pct, base_f1_1pct)]

        rows.append({
            "Stage": stage_name,
            "Cols": len(cols),
            "Acc (Orig)": format_metric([m["Accuracy"] for m in orig_list]),
            "F1 (Orig 42.8%)": format_metric(f1_orig),
            "ΔF1 (Orig)": format_metric(drops_orig, signed=True),
            "PR-AUC (Orig)": format_metric([m["PR-AUC"] for m in orig_list]),
            "Prec (1% Pos)": format_metric([m["Precision"] for m in pct1_list]),
            "F1 (1% Pos)": format_metric(f1_1pct),
            "ΔF1 (1% Pos)": format_metric(drops_1pct, signed=True),
            "PR-AUC (1% Pos)": format_metric([m["PR-AUC"] for m in pct1_list]),
        })

    result_df = pd.DataFrame(rows)
    suffix = f"K={len(splits)} seeds, Mean ± 95% CI" if len(splits) >= 2 else f"seed={splits[0][0]}"
    print(f"\n================ FEATURE ABLATION: {model_name} (eTLD+1 SPLIT, {suffix}) ================")
    print(result_df.to_string(index=False))
    return result_df


def run_single_leakage_audit(
    splits: list[tuple[int, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_LEAKAGE_COLS:
        metrics_list = []
        for seed, train_df, test_df in splits:
            X_train, y_train = extract_features_and_labels(train_df, [col])
            X_test, y_test = extract_features_and_labels(test_df, [col])
            model = build_logistic_regression_pipeline(seed)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics_list.append(evaluate_predictions(y_test, y_pred, y_prob))

        rows.append({
            "Single Leaked Feature": col,
            **{m: format_metric([rec[m] for rec in metrics_list]) for m in CORE_METRICS},
        })

    audit_df = pd.DataFrame(rows)
    suffix = f"K={len(splits)} seeds, Mean ± 95% CI" if len(splits) >= 2 else f"seed={splits[0][0]}"
    print(f"\n================ SINGLE-FEATURE LEAKAGE AUDIT (1 COLUMN ONLY, {suffix}) ================")
    print(audit_df.to_string(index=False))
    return audit_df


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    seeds = normalize_seeds(SEEDS)
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"2. Splitting dataset by eTLD+1 across K={len(seeds)} seed(s): {list(seeds)}...")
    splits = []
    for s in seeds:
        train_etld1, test_etld1 = etld1_split(
            df,
            test_size=TEST_SIZE,
            random_state=s,
            url_col="URL",
            n_candidates=100,
        )
        assert_no_etld1_overlap(train_etld1, test_etld1, url_col="URL")
        splits.append((s, train_etld1, test_etld1))

    print("\n3. Running Feature Ablation on Logistic Regression...")
    run_feature_ablation(
        "LOGISTIC REGRESSION",
        build_logistic_regression_pipeline,
        splits,
    )

    print("\n4. Running Feature Ablation on Gradient Boosting...")
    run_feature_ablation(
        "GRADIENT BOOSTING",
        build_gb_model,
        splits,
    )

    print("\n5. Auditing each leaked feature individually...")
    run_single_leakage_audit(splits)


if __name__ == "__main__":
    main()
