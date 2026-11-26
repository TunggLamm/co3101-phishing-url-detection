"""Run all locked experiments once and export:
  - ``results/experiment_run.log`` (full console output)
  - ``results/metrics.json`` (compact numbers for the 4 report figures)
  - ``results/figures/*.png`` (unless ``--skip-plots`` is passed)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import run_ablation as ablation_mod
from src.data_split import assert_no_etld1_overlap, etld1_split, random_split
from src.error_analysis import (
    RANDOM_STATE as ERROR_SEED,
    build_error_frame,
    print_error_counts_and_overlap,
    print_feature_profile,
    print_template_breakdown,
    print_top_error_examples,
)
from src.evaluation import DEFAULT_SEEDS, REALISTIC_POSITIVE_RATES, evaluate_model, normalize_seeds
from src.features import DATA_PATH, SELECTED_LEXICAL_FEATURES, select_lexical_features
from src.one_line_rule import build_one_line_rule_model, extract_url_and_labels, print_rule_assumption_check
from src.train_gb import build_gb_model, print_feature_importances
from src.train_lr import build_logistic_regression_pipeline, print_feature_weights
from src.utils import compute_confidence_interval

RESULTS_DIR = PROJECT_DIR / "results"
TEST_SIZE = 0.2
BASELINE_SEEDS = normalize_seeds(DEFAULT_SEEDS)


def parse_metric(text: object) -> list[float]:
    """Parse ``'0.9876 ± 0.0028'`` or ``'0.9957'`` into ``[mean, half_width]``."""
    parts = [p.strip() for p in str(text).split("±", maxsplit=1)]
    return [float(parts[0]), float(parts[1]) if len(parts) == 2 else 0.0]


def _mean_hw(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return round(float(values[0]), 4), 0.0
    mean, lo, hi = compute_confidence_interval(values)
    return round(mean, 4), round(0.5 * (hi - lo), 4)


def run_all_exports(generate_plots: bool = True) -> dict:
    """Run all locked experiments and save log, metrics.json, and figures."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ablation_seeds = normalize_seeds(ablation_mod.SEEDS)
    all_seeds = sorted(set(BASELINE_SEEDS) | set(ablation_seeds) | {ERROR_SEED})
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buf = io.StringIO()

    with redirect_stdout(buf):
        print(f"=== CO3101-C1 LOCKED EXPERIMENT RUN ({run_ts}) ===")
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df):,} rows x {len(df.columns)} columns.")
        print(f"Columns (verified no timestamp): {df.columns.tolist()}")

        rand_splits = {s: random_split(df, test_size=TEST_SIZE, random_state=s, label_col="label") for s in all_seeds}
        etld_splits = {}
        for s in all_seeds:
            tr, te = etld1_split(df, test_size=TEST_SIZE, random_state=s, url_col="URL", n_candidates=100)
            assert_no_etld1_overlap(tr, te, url_col="URL")
            etld_splits[s] = (tr, te)

        # 1. Three mandatory baselines (K=5 seeds, Random vs eTLD+1)
        baselines_data, etld_models = [], {}
        for label, builder, extractor in [
            ("One-Line Rule", build_one_line_rule_model, extract_url_and_labels),
            ("Logistic Regression", build_logistic_regression_pipeline, select_lexical_features),
            ("Gradient Boosting", build_gb_model, select_lexical_features),
        ]:
            for split_name, split_map, check_disjoint in [
                ("Random Split", rand_splits, False),
                ("eTLD+1 Split", etld_splits, True),
            ]:
                res = evaluate_model(
                    df,
                    split_name=f"{label.upper()} - {split_name.upper()}",
                    split_fn=lambda _d, s, m=split_map: m[s],
                    model_builder=builder,
                    feature_extractor=extractor,
                    seeds=BASELINE_SEEDS,
                    target_rates=REALISTIC_POSITIVE_RATES,
                    assert_etld1_disjoint=check_disjoint,
                )
                if split_name == "eTLD+1 Split":
                    etld_models[label] = res["models"]
                for _, row in res["rate_summary_df"].iterrows():
                    raw_rate = str(row["Target Pos Rate"])
                    baselines_data.append({
                        "model": label,
                        "split": split_name,
                        "rate_key": "original" if "Original" in raw_rate else raw_rate,
                        "Precision": parse_metric(row["Precision"]),
                        "F1-score": parse_metric(row["F1-score"]),
                        "PR-AUC": parse_metric(row["PR-AUC"]),
                    })

        print_rule_assumption_check(df)
        print_feature_weights(etld_models["Logistic Regression"])
        print_feature_importances(etld_models["Gradient Boosting"])

        # 2. Feature ablation & single-feature leakage audit
        abl_splits = [(s, *etld_splits[s]) for s in ablation_seeds]
        abl_lr = ablation_mod.run_feature_ablation("LOGISTIC REGRESSION", build_logistic_regression_pipeline, abl_splits)
        abl_gb = ablation_mod.run_feature_ablation("GRADIENT BOOSTING", build_gb_model, abl_splits)
        leak_df = ablation_mod.run_single_leakage_audit(abl_splits)

        # 3. Cross-model FP/FN error analysis (reusing returned dicts directly)
        error_df = build_error_frame(*etld_splits[ERROR_SEED])
        err_summary = print_error_counts_and_overlap(error_df)
        err_summary["fn_traits_pct"] = print_template_breakdown(error_df)
        print_feature_profile(error_df, "LOGISTIC REGRESSION", "lr_pred")
        print_feature_profile(error_df, "GRADIENT BOOSTING", "gb_pred")
        print_top_error_examples(error_df, "LOGISTIC REGRESSION", "lr_pred", "lr_prob")
        print_top_error_examples(error_df, "GRADIENT BOOSTING", "gb_pred", "gb_prob")

    (RESULTS_DIR / "experiment_run.log").write_text(buf.getvalue(), encoding="utf-8")

    gb_imps = np.vstack([m.feature_importances_ for m in etld_models["Gradient Boosting"]])
    lr_coefs = np.vstack([m.named_steps["clf"].coef_[0] for m in etld_models["Logistic Regression"]])
    metrics = {
        "meta": {
            "run_timestamp": run_ts,
            "baseline_seeds": list(BASELINE_SEEDS),
            "ablation_seeds": list(ablation_seeds),
            "error_seed": ERROR_SEED,
        },
        "baselines": baselines_data,
        "ablation": {
            name: [
                {
                    "stage": row["Stage"],
                    "cols": int(row["Cols"]),
                    "f1_orig": parse_metric(row["F1 (Orig 42.8%)"]),
                    "f1_1pct": parse_metric(row["F1 (1% Pos)"]),
                }
                for _, row in tbl.iterrows()
            ]
            for name, tbl in [("Logistic Regression", abl_lr), ("Gradient Boosting", abl_gb)]
        },
        "single_leakage": [
            {"feature": r["Single Leaked Feature"], "f1": parse_metric(r["F1-score"]), "pr_auc": parse_metric(r["PR-AUC"])}
            for _, r in leak_df.iterrows()
        ],
        "feature_importance": {
            "gb": [
                {"feature": f, "mean": m, "hw": hw, "share_pct": round(m * 100.0, 2)}
                for i, f in enumerate(SELECTED_LEXICAL_FEATURES)
                for m, hw in [_mean_hw(gb_imps[:, i])]
            ],
            "lr": [
                {"feature": f, "mean": m, "hw": hw}
                for i, f in enumerate(SELECTED_LEXICAL_FEATURES)
                for m, hw in [_mean_hw(lr_coefs[:, i])]
            ],
        },
        "errors": err_summary,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved results/experiment_run.log and results/metrics.json.")

    if generate_plots:
        from src.plot_figures import generate_all_figures

        for path in generate_all_figures():
            print(f"Saved {path.relative_to(PROJECT_DIR)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked experiments and export log, metrics.json, and figures.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating PNG figures.")
    run_all_exports(generate_plots=not parser.parse_args().skip_plots)


if __name__ == "__main__":
    main()
