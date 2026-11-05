""" Train and evaluate the Gradient Boosting baseline model on 10 lexical features """

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import (
    assert_no_etld1_overlap,
    etld1_split,
    random_split,
)
from src.features import (
    DATA_PATH,
    FEATURE_DESCRIPTIONS,
    SELECTED_LEXICAL_FEATURES,
    select_lexical_features,
)
from src.utils import (
    adjust_positive_rate,
    compute_pr_auc,
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
REALISTIC_POSITIVE_RATES = (0.10, 0.05, 0.01)

def build_gb_model() -> GradientBoostingClassifier:
    """
    Build the Gradient Boosting classifier
    Tree-based models split on feauture thresholds and do not require StandardScaler
    """
    return GradientBoostingClassifier (
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
    )

def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict:
    """Compute core classification metrics using custom compute_pr_auc."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "PR-AUC": compute_pr_auc(y_true, y_prob, pos_label=1),
    }

def evaluate_realistic_positive_rates(
    split_name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_rates: tuple[float, ...] = REALISTIC_POSITIVE_RATES,
) -> pd.DataFrame:
    """
    Evaluate model metrics when lowering the test positive rate to realistic prevalence
    levels (e.g., 10%, 5%, 1%) while keeping 100% of positive phishing samples intact.
    """
    eval_df = pd.DataFrame({
        "label": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    })

    base_rate = float((eval_df["label"] == 1).mean())
    rows = []

    base_metrics = evaluate_predictions(eval_df["label"], eval_df["y_pred"], eval_df["y_prob"])
    rows.append({
        "Target Pos Rate": f"{base_rate:.2%} (Original)",
        "Total Rows": len(eval_df),
        "Pos Rows (1)": int((eval_df["label"] == 1).sum()),
        "Neg Rows (0)": int((eval_df["label"] == 0).sum()),
        **{k: f"{v:.4f}" for k, v in base_metrics.items()},
    })

    for rate in target_rates:
        adj_df = adjust_positive_rate(
            eval_df,
            target_rate=rate,
            label_col="label",
            pos_label=1,
            random_state=RANDOM_STATE,
        )
        adj_metrics = evaluate_predictions(adj_df["label"], adj_df["y_pred"], adj_df["y_prob"])
        actual_rate = float((adj_df["label"] == 1).mean())
        rows.append({
            "Target Pos Rate": f"{actual_rate:.2%}",
            "Total Rows": len(adj_df),
            "Pos Rows (1)": int((adj_df["label"] == 1).sum()),
            "Neg Rows (0)": int((adj_df["label"] == 0).sum()),
            **{k: f"{v:.4f}" for k, v in adj_metrics.items()},
        })

    rate_df = pd.DataFrame(rows)
    print(f"\n--- Realistic Positive-Rate Evaluation ({split_name}) ---")
    print(rate_df.to_string(index=False))
    return rate_df

def train_and_evaluate(
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[GradientBoostingClassifier, dict]:
    """
    Extract the 10 lexical features, train Gradient Boosting, and compute evaluation metrics
    Standardized labels from select_lexical_features: 1 = Phishing (positive), 0 = Legitimate (negative)
    """
    X_train, y_train = select_lexical_features(train_df)
    X_test, y_test = select_lexical_features(test_df)

    model = build_gb_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    core_metrics = evaluate_predictions(y_test, y_pred, y_prob)
    metrics = {
        "Train Rows": len(X_train),
        "Test Rows": len(X_test),
        **core_metrics,
    }

    print(f"\n================ {split_name} ================")
    print(f"Train rows : {len(X_train):,} | Test rows : {len(X_test):,}")
    print(f"Test labels: Legitimate (0) = {(y_test == 0).sum():,} | Phishing (1) = {(y_test == 1).sum():,}")
    print(f"  + Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  + Precision : {metrics['Precision']:.4f}")
    print(f"  + Recall    : {metrics['Recall']:.4f}")
    print(f"  + F1-score  : {metrics['F1-score']:.4f}")
    print(f"  + PR-AUC    : {metrics['PR-AUC']:.4f}")

    print("\n--- Confusion Matrix [TN FP / FN TP] ---")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Legitimate (0)", "Phishing (1)"],
            digits=4,
            zero_division=0,
        )
    )

    evaluate_realistic_positive_rates(split_name, y_test, y_pred, y_prob)

    return model, metrics

def print_features_importances(model: GradientBoostingClassifier) -> None:
    """ Print tree-based feature importances of the 10 lexical features sorted descending """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Description": [FEATURE_DESCRIPTIONS[col] for col in SELECTED_LEXICAL_FEATURES],
        "Importance": [round(imp, 4) for imp in importances],
        "Share (%)": [f"{imp * 100:.2f}%" for imp in importances],
    }, index=SELECTED_LEXICAL_FEATURES).sort_values(by="Importance", ascending=False)

    print("\n--- Gradient Boosting Feature Importances (eTLD+1 Split Model) ---")
    print(importance_df.to_string())

def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    #Branch 1: Stratified Random Split
    print("2. Splitting dataset and training Gradient Boosting on Random Split...")
    train_random, test_random = random_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        label_col="label",
    )
    _, metrics_random = train_and_evaluate(
        "GRADIENT BOOSTING - RANDOM SPLIT (10 LEXICAL FEATURES)",
        train_random,
        test_random,
    )

    #Branch 2: eTLD+1 Domain-Disjoint Split
    print("\n3. Splitting dataset and training Gradient Boosting on eTLD+1 Split...")
    train_etld1, test_etld1 = etld1_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        url_col="URL",
        n_candidates=100,
    )
    assert_no_etld1_overlap(train_etld1, test_etld1, url_col="URL")
    model_etld1, metrics_etld1 = train_and_evaluate(
        "GRADIENT BOOSTING - eTLD+1 SPLIT (10 LEXICAL FEATURES)",
        train_etld1,
        test_etld1,
    )

    #Summary Comparision table
    comparison_df = pd.DataFrame(
        [metrics_random, metrics_etld1],
        index=["GB - Random Split", "GB - eTLD+1 Split"],
    )
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "PR-AUC"]:
        comparison_df[col] = comparison_df[col].map(lambda v: f"{v:.4f}")

    print("\n================ GRADIENT BOOSTING COMPARISON SUMMARY ================")
    print(comparison_df.to_string())
    print("======================================================================")

    print_features_importances(model_etld1)

if __name__ == "__main__":
    main()