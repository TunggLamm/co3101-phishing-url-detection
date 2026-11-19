import sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import assert_no_etld1_overlap, etld1_split
from src.features import (
    DATA_PATH,
    NUMERIC_LEAKAGE_COLS,
    extract_features_and_labels,
    get_ablation_feature_sets,
)
from src.train_gb import build_gb_model
from src.train_lr import build_logistic_regression_pipeline, evaluate_predictions
from src.utils import adjust_positive_rate

TEST_SIZE = 0.2
RANDOM_STATE = 42
REALISTIC_POSITIVE_RATES = 0.01

def evaluate_at_rate(
    y_test: pd.Series,
    y_pred,
    y_prob,
    target_rate: float | None = None,
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
        target_rate = target_rate,
        label_col = "label",
        pos_label = 1,
        random_state = RANDOM_STATE,
    )
    return evaluate_predictions(adj_df["label"], adj_df["y_pred"], adj_df["y_prob"])

def run_feature_ablation(
    model_name: str,
    model_builder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_set = get_ablation_feature_sets(train_df)
    rows = []
    baseline_f1_orig = None
    baseline_f1_1pct = None

    for stage_name, cols in feature_set.items():
        X_train, y_train = extract_features_and_labels(train_df, cols)
        X_test, y_test = extract_features_and_labels(test_df, cols)
        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        m_orig = evaluate_at_rate(y_test, y_pred, y_prob, target_rate=None)
        m_1pct = evaluate_at_rate(y_test, y_pred, y_prob, target_rate=REALISTIC_POSITIVE_RATES)

        if baseline_f1_orig is None:
            baseline_f1_orig = m_orig["F1-score"]
            baseline_f1_1pct = m_1pct["F1-score"]
        
        drop_orig = m_orig["F1-score"] - baseline_f1_orig
        drop_1pct = m_1pct["F1-score"] - baseline_f1_1pct

        rows.append({
            "Stage": stage_name,
            "Cols": len(cols),
            "Acc (Orig)": f"{m_orig['Accuracy']:.4f}",
            "F1 (Orig 42.8%)": f"{m_orig['F1-score']:.4f}",
            "ΔF1 (Orig)": f"{drop_orig:+.4f}",
            "PR-AUC (Orig)": f"{m_orig['PR-AUC']:.4f}",
            "Prec (1% Pos)": f"{m_1pct['Precision']:.4f}",
            "F1 (1% Pos)": f"{m_1pct['F1-score']:.4f}",
            "ΔF1 (1% Pos)": f"{drop_1pct:+.4f}",
            "PR-AUC (1% Pos)": f"{m_1pct['PR-AUC']:.4f}",
        })

    result_df = pd.DataFrame(rows)
    print(f"\n================ FEATURE ABLATION: {model_name} (eTLD+1 SPLIT) ================")
    print(result_df.to_string(index=False))
    return result_df

def run_single_leakage_audit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_LEAKAGE_COLS:
        X_train, y_train = extract_features_and_labels(train_df, [col])
        X_test, y_test = extract_features_and_labels(test_df, [col])
        model = build_logistic_regression_pipeline()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = evaluate_predictions(y_test, y_pred, y_prob)

        rows.append({
            "Single Leaked Feature": col,
            "Accuracy": f"{m['Accuracy']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall": f"{m['Recall']:.4f}",
            "F1-score": f"{m['F1-score']:.4f}",
            "PR-AUC": f"{m['PR-AUC']:.4f}",
        })

    audit_df = pd.DataFrame(rows)
    print("\n================ SINGLE-FEATURE LEAKAGE AUDIT (1 COLUMN ONLY) ================")
    print(audit_df.to_string(index=False))
    return audit_df

def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)
    print("2. Splitting dataset by eTLD+1 (domain-disjoint split)...")
    train_etld1, test_etld1 = etld1_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        url_col="URL",
        n_candidates=100,
    )
    assert_no_etld1_overlap(train_etld1, test_etld1, url_col="URL")

    print("\n3. Running Feature Ablation on Logistic Regression...")
    run_feature_ablation(
        "LOGISTIC REGRESSION",
        build_logistic_regression_pipeline,
        train_etld1,
        test_etld1,
    )

    print("\n4. Running Feature Ablation on Gradient Boosting...")
    run_feature_ablation(
        "GRADIENT BOOSTING",
        build_gb_model,
        train_etld1,
        test_etld1,
    )

    print("\n5. Auditing each leaked feature individually...")
    run_single_leakage_audit(train_etld1, test_etld1)

if __name__ == "__main__":
    main()