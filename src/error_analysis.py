"""
Error Analysis for the 3 locked baselines (One-Line Rule, Logistic Regression,
Gradient Boosting) on the eTLD+1 domain-disjoint split (seed=42).

Inspects:
1. FP / FN counts, error rates (FPR, FNR), and cross-model error overlap.
2. Structural template breakdown (https://www. prefix, root-only URLs, digits, queries).
3. Feature mean profile comparison (TN vs FP, TP vs FN) across the 10 lexical features.
4. High-confidence concrete URL examples for manual inspection and Error Taxonomy.
"""
import sys 
from pathlib import Path 

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import assert_no_etld1_overlap, etld1_split
from src.features import (
    DATA_PATH,
    SELECTED_LEXICAL_FEATURES,
    select_lexical_features,
)
from src.one_line_rule import RULE_PREFIX, build_one_line_rule_model
from src.train_gb import build_gb_model
from src.train_lr import build_logistic_regression_pipeline

TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_EXAMPLES = 10

def build_error_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Train the locked baselines on eTLD+1 train split and collect test predictions."""
    X_train, y_train = select_lexical_features(train_df)
    X_test, y_test = select_lexical_features(test_df)

    # 1. One-Line Rule
    rule_model = build_one_line_rule_model(RANDOM_STATE)
    rule_pred = rule_model.predict(test_df[["URL"]])

    # 2. Logistic Regression (10 lexical features)
    lr_model = build_logistic_regression_pipeline(RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    # 3. Gradient Boosting (10 lexical features)
    gb_model = build_gb_model(RANDOM_STATE)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_prob = gb_model.predict_proba(X_test)[:, 1]

    analysis_df = test_df[["URL", "Domain", "TLD"] + SELECTED_LEXICAL_FEATURES].copy()
    analysis_df["y_true"] = y_test.to_numpy()
    analysis_df["rule_pred"] = rule_pred
    analysis_df["lr_pred"] = lr_pred
    analysis_df["lr_prob"] = lr_prob
    analysis_df["gb_pred"] = gb_pred
    analysis_df["gb_prob"] = gb_prob

    url_str = analysis_df["URL"].fillna("").astype(str)
    analysis_df["starts_https_www"] = url_str.str.startswith(RULE_PREFIX)
    analysis_df["len_diff"] = analysis_df["URLLength"] - analysis_df["DomainLength"]
    analysis_df["is_root_len_diff_7"] = analysis_df["len_diff"] == 7
    analysis_df["has_digits"] = analysis_df["DegitRatioInURL"] > 0
    analysis_df["has_query"] = (
        (analysis_df["NoOfQMarkInURL"] > 0) | (analysis_df["NoOfEqualsInURL"] > 0)
    )
    return analysis_df

def print_error_counts_and_overlap(df: pd.DataFrame) -> None:
    models = [
        ("One-Line Rule", "rule_pred"),
        ("Logistic Regression (10 feats)", "lr_pred"),
        ("Gradient Boosting (10 feats)", "gb_pred"),
    ]
    rows = []
    for name, col in models:
        tn = int(((df["y_true"] == 0) & (df[col] == 0)).sum())
        fp = int(((df["y_true"] == 0) & (df[col] == 1)).sum())
        fn = int(((df["y_true"] == 1) & (df[col] == 0)).sum())
        tp = int(((df["y_true"] == 1) & (df[col] == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append({
            "Model": name,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "FPR (%)": f"{fpr * 100:.2f}%",
            "FNR (%)": f"{fnr * 100:.2f}%",
        })
    
    print("\n================ 1. ERROR COUNTS SUMMARY (eTLD+1 TEST SET) ================")
    print(pd.DataFrame(rows).to_string(index=False))
    rule_fn = (df["y_true"] == 1) & (df["rule_pred"] == 0)
    lr_fn = (df["y_true"] == 1) & (df["lr_pred"] == 0)
    gb_fn = (df["y_true"] == 1) & (df["gb_pred"] == 0)

    lr_fp = (df["y_true"] == 0) & (df["lr_pred"] == 1)
    gb_fp = (df["y_true"] == 0) & (df["gb_pred"] == 1)

    gb_fn_total = int(gb_fn.sum())
    gb_fp_total = int(gb_fp.sum())

    print("\n--- Cross-Model Error Overlap ---")
    print(
        f"FN overlap (GB ∩ One-Line Rule) : {int((gb_fn & rule_fn).sum()):,} / {gb_fn_total:,} "
        f"({100 * (gb_fn & rule_fn).sum() / gb_fn_total:.2f}% of GB FN)"
    )
    print(
        f"FN overlap (GB ∩ LR)            : {int((gb_fn & lr_fn).sum()):,} / {gb_fn_total:,} "
        f"({100 * (gb_fn & lr_fn).sum() / gb_fn_total:.2f}% of GB FN)"
    )
    print(
        f"FN overlap (All 3 Models)       : {int((gb_fn & lr_fn & rule_fn).sum()):,} / {gb_fn_total:,} "
        f"({100 * (gb_fn & lr_fn & rule_fn).sum() / gb_fn_total:.2f}% of GB FN)"
    )
    print(
        f"FP overlap (GB ∩ LR)            : {int((gb_fp & lr_fp).sum()):,} / {gb_fp_total:,} "
        f"({100 * (gb_fp & lr_fp).sum() / gb_fp_total:.2f}% of GB FP)"
    )

def print_template_breakdown(df: pd.DataFrame) -> None:
    models = [
        ("One-Line Rule", "rule_pred"),
        ("Logistic Regression", "lr_pred"),
        ("Gradient Boosting", "gb_pred"),
    ]
    fn_rows = []
    for name, col in models: 
        sub = df[(df["y_true"] == 1) & (df[col] == 0)]
        fn_rows.append({
            "Model (FN Group)": name,
            "FN Count": len(sub),
            "% https://www.": f"{sub['starts_https_www'].mean() * 100:.2f}%",
            "% LenDiff == 7": f"{sub['is_root_len_diff_7'].mean() * 100:.2f}%",
            "% Zero Digits": f"{(~sub['has_digits']).mean() * 100:.2f}%",
            "% No Query (?/=)": f"{(~sub['has_query']).mean() * 100:.2f}%",
        })
    print("\n================ 2A. FN STRUCTURAL BREAKDOWN (MISSED PHISHING) ================")
    print(pd.DataFrame(fn_rows).to_string(index=False))
    legit_all = df[df["y_true"] == 0]
    fp_rows = [{
        "Group (Legitimate)": "All Legitimate (Baseline)",
        "Count": len(legit_all),
        "% Has Digits": f"{legit_all['has_digits'].mean() * 100:.2f}%",
        "Avg URLLength": f"{legit_all['URLLength'].mean():.2f}",
        "Avg DigitRatio": f"{legit_all['DegitRatioInURL'].mean():.4f}",
        "Avg SpecialRatio": f"{legit_all['SpacialCharRatioInURL'].mean():.4f}",
        "Avg SubDomain": f"{legit_all['NoOfSubDomain'].mean():.2f}",
    }]
    for name, col in models[1:]:  # Bỏ qua One-Line Rule vì FP = 0
        sub = df[(df["y_true"] == 0) & (df[col] == 1)]
        fp_rows.append({
            "Group (Legitimate)": f"{name} (FP)",
            "Count": len(sub),
            "% Has Digits": f"{sub['has_digits'].mean() * 100:.2f}%",
            "Avg URLLength": f"{sub['URLLength'].mean():.2f}",
            "Avg DigitRatio": f"{sub['DegitRatioInURL'].mean():.4f}",
            "Avg SpecialRatio": f"{sub['SpacialCharRatioInURL'].mean():.4f}",
            "Avg SubDomain": f"{sub['NoOfSubDomain'].mean():.2f}",
        })
    
    print("\n================ 2B. FP STRUCTURAL BREAKDOWN (FALSE ALARMS) ================")
    print(pd.DataFrame(fp_rows).to_string(index=False))

def print_feature_profile(
    df: pd.DataFrame,
    model_name: str,
    pred_col:str,
) -> None:
    tn = df[(df["y_true"] == 0) & (df[pred_col] == 0)][SELECTED_LEXICAL_FEATURES].mean()
    fp = df[(df["y_true"] == 0) & (df[pred_col] == 1)][SELECTED_LEXICAL_FEATURES].mean()
    tp = df[(df["y_true"] == 1) & (df[pred_col] == 1)][SELECTED_LEXICAL_FEATURES].mean()
    fn = df[(df["y_true"] == 1) & (df[pred_col] == 0)][SELECTED_LEXICAL_FEATURES].mean()

    profile_df = pd.DataFrame({
        "TN (Legit OK)": tn.round(4),
        "FP (Legit->Phish)": fp.round(4),
        "TP (Phish OK)": tp.round(4),
        "FN (Phish->Legit)": fn.round(4),
    })

    print(f"\n================ 3. FEATURE PROFILE: {model_name} ================")
    print(profile_df.to_string())

def print_top_error_examples(
    df: pd.DataFrame,
    model_name: str,
    pred_col: str,
    prob_col: str,
    top_n: int = TOP_N_EXAMPLES,
) -> None:
    display_cols = [
        "URL_short",
        prob_col,
        "starts_https_www",
        "URLLength",
        "DomainLength",
        "DegitRatioInURL",
        "LetterRatioInURL",
        "SpacialCharRatioInURL",
    ]

    work = df.copy()
    work["URL_short"] = work["URL"].astype(str).str.slice(0, 65)

    fp_df = (
        work[(work["y_true"] == 0) & (work[pred_col] == 1)]
        .sort_values(by=prob_col, ascending=False)
        .head(top_n)
    )
    fn_df = (
        work[(work["y_true"] == 1) & (work[pred_col] == 0)]
        .sort_values(by=prob_col, ascending=True)
        .head(top_n)
    )

    print(f"\n--- Top {top_n} High-Confidence FALSE POSITIVES ({model_name}) ---")
    print(fp_df[display_cols].to_string(index=False))

    print(f"\n--- Top {top_n} High-Confidence FALSE NEGATIVES ({model_name}) ---")
    print(fn_df[display_cols].to_string(index=False))

def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    print("1. Loading PhiUSIIL dataset...")
    df = pd.read_csv(DATA_PATH)

    print("2. Splitting dataset by eTLD+1 (seed=42)...")
    train_etld1, test_etld1 = etld1_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        url_col="URL",
        n_candidates=100,
    )
    assert_no_etld1_overlap(train_etld1, test_etld1, url_col="URL")

    print("3. Training baselines and collecting test-set predictions...")
    error_df = build_error_frame(train_etld1, test_etld1)

    print_error_counts_and_overlap(error_df)
    print_template_breakdown(error_df)
    print_feature_profile(error_df, "LOGISTIC REGRESSION", "lr_pred")
    print_feature_profile(error_df, "GRADIENT BOOSTING", "gb_pred")
    print_top_error_examples(error_df, "LOGISTIC REGRESSION", "lr_pred", "lr_prob")
    print_top_error_examples(error_df, "GRADIENT BOOSTING", "gb_pred", "gb_prob")
    rule_fn = error_df[(error_df["y_true"] == 1) & (error_df["rule_pred"] == 0)].head(TOP_N_EXAMPLES).copy()
    rule_fn["URL_short"] = rule_fn["URL"].astype(str).str.slice(0, 70)
    print(f"\n--- Sample {TOP_N_EXAMPLES} FALSE NEGATIVES (One-Line Rule: Phishing starting with https://www.) ---")
    print(rule_fn[["URL_short", "URLLength", "DomainLength", "len_diff", "has_digits"]].to_string(index=False))

if __name__ == "__main__":
    main()