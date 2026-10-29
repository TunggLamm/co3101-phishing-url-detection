import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

TEST_SIZE = 0.2
RANDOM_STATE = 42

def build_logistic_regression_pipeline() -> Pipeline:
    """ Build a standard machine learning pipeline with StandardScaler and LogisticRegression """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

def train_and_evaluate(
        split_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Extract the 10 lexical features, train Logistic Regression, and compute evaluation metrics.
    Standardized labels from select_lexical_features: 1 = Phishing (Positive), 0 = Legitimate (Negative)
    """
    X_train, y_train = select_lexical_features(train_df)
    X_test, y_test = select_lexical_features(test_df)

    model = build_logistic_regression_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Train Rows": len(X_train),
        "Test Rows": len(X_test),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "PR-AUC": average_precision_score(y_test, y_prob),
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

    return model, metrics

def print_feature_weights(model: Pipeline) -> None:
    """ Print standardized weights of the 10 lexical features sorted by magnitude """
    coefs = model.named_steps["clf"].coef_[0]
    weights_df = pd.DataFrame({
        "Description": [FEATURE_DESCRIPTIONS[col] for col in SELECTED_LEXICAL_FEATURES],
        "Weight (w)": [round(c, 4) for c in coefs],
        "|Weight|": [round(abs(c), 4) for c in coefs],
        "Direction": [
            "Phishing (+)" if c > 0 else "Legitimate (-)"
            for c in coefs
        ],
    }, index=SELECTED_LEXICAL_FEATURES).sort_values(by="|Weight|", ascending=False)

    print("\n--- Standardized Feature Weights (eTLD+1 Split Model) ---")
    print(weights_df.to_string())

def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    # Nhánh 1: Chia ngẫu nhiên (Random Split) từ module data_split của Hiếu
    print("2. Splitting dataset and training on Random Split...")
    train_random, test_random = random_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        label_col="label",
    )
    _, metrics_random = train_and_evaluate(
        "LOGISTIC REGRESSION - RANDOM SPLIT (10 LEXICAL FEATURES)",
        train_random,
        test_random,
    )

    # Nhánh 2: Chia tách rời theo tên miền (eTLD+1 Split) từ module data_split của Hiếu
    print("\n3. Splitting dataset and training on eTLD+1 Split...")
    train_etld1, test_etld1 = etld1_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        url_col="URL",
        n_candidates=100,
    )
    assert_no_etld1_overlap(train_etld1, test_etld1, url_col="URL")
    model_etld1, metrics_etld1 = train_and_evaluate(
        "LOGISTIC REGRESSION - eTLD+1 SPLIT (10 LEXICAL FEATURES)",
        train_etld1,
        test_etld1,
    )

    # Bảng tổng hợp đối chứng 2 cách chia
    comparison_df = pd.DataFrame(
        [metrics_random, metrics_etld1],
        index=["Random Split", "eTLD+1 Split"],
    )
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "PR-AUC"]:
        comparison_df[col] = comparison_df[col].map(lambda v: f"{v:.4f}")

    print("\n================ BASELINE COMPARISON SUMMARY ================")
    print(comparison_df.to_string())
    print("================================================================")

    # In độ quan trọng của 10 đặc trưng
    print_feature_weights(model_etld1)

if __name__ == "__main__":
    main()