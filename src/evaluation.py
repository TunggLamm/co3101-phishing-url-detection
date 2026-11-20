"""Single-seed and multi-seed model evaluation with optional Student-t CIs."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_split import assert_no_etld1_overlap
from src.features import select_lexical_features
from src.utils import (
    adjust_positive_rate,
    compute_confidence_interval,
    compute_pr_auc,
)

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_CONFIDENCE: float = 0.95
REALISTIC_POSITIVE_RATES: tuple[float, ...] = (0.10, 0.05, 0.01)
CORE_METRICS: tuple[str, ...] = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "PR-AUC",
)


def normalize_seeds(seeds: int | Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a single int seed or a sequence of distinct ints."""
    if isinstance(seeds, bool):
        raise ValueError(f"Seed must be an integer; got {seeds!r}.")
    if isinstance(seeds, (int, np.integer)):
        return (int(seeds),)

    validated: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"All seeds must be integers; got {seed!r}.")
        validated.append(int(seed))

    if not validated:
        raise ValueError("At least 1 seed is required.")
    if len(set(validated)) != len(validated):
        raise ValueError(f"Seeds must be distinct; got {validated}.")

    return tuple(validated)


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    y_prob: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute core classification metrics using custom compute_pr_auc."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "F1-score": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "PR-AUC": float(compute_pr_auc(y_true, y_prob, pos_label=1)),
    }


def format_metric(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> str:
    """Format a single value as ``0.xxxx`` or >= 2 values as ``Mean ± CI``."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return f"{float(arr[0]):.4f}"

    mean, ci_low, ci_high = compute_confidence_interval(
        arr,
        confidence=confidence,
    )
    half_width = 0.5 * (ci_high - ci_low)
    return f"{mean:.4f} ± {half_width:.4f}"


def evaluate_model(
    df: pd.DataFrame,
    *,
    split_name: str,
    split_fn: Callable[[pd.DataFrame, int], tuple[pd.DataFrame, pd.DataFrame]],
    model_builder: Callable[[int], Any],
    seeds: int | Sequence[int] = DEFAULT_SEEDS,
    confidence: float = DEFAULT_CONFIDENCE,
    target_rates: Sequence[float] = REALISTIC_POSITIVE_RATES,
    assert_etld1_disjoint: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a model across 1 seed or K seeds (reporting Mean ± CI).

    Returns a dict with ``seeds``, ``models``, ``per_seed_df``, ``summary``,
    and ``rate_summary_df``.
    """
    validated_seeds = normalize_seeds(seeds)
    is_multi = len(validated_seeds) >= 2
    per_seed_rows: list[dict[str, Any]] = []
    rate_records: list[dict[str, Any]] = []
    models: list[Any] = []

    for seed in validated_seeds:
        train_df, test_df = split_fn(df, seed)
        if assert_etld1_disjoint:
            assert_no_etld1_overlap(train_df, test_df, url_col="URL")

        X_train, y_train = select_lexical_features(train_df)
        X_test, y_test = select_lexical_features(test_df)

        model = model_builder(seed)
        model.fit(X_train, y_train)
        models.append(model)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        base_metrics = evaluate_predictions(y_test, y_pred, y_prob)
        base_rate = float((y_test == 1).mean())
        per_seed_rows.append(
            {
                "Seed": seed,
                "Train Rows": len(X_train),
                "Test Rows": len(X_test),
                **base_metrics,
            }
        )
        rate_records.append(
            {
                "rate_key": "original",
                "pos_rate": base_rate,
                "total_rows": len(y_test),
                "pos_rows": int((y_test == 1).sum()),
                "neg_rows": int((y_test == 0).sum()),
                **base_metrics,
            }
        )

        eval_df = pd.DataFrame(
            {"label": y_test.to_numpy(), "y_pred": y_pred, "y_prob": y_prob}
        )
        for rate in target_rates:
            adj_df = adjust_positive_rate(
                eval_df,
                target_rate=rate,
                label_col="label",
                pos_label=1,
                random_state=seed,
            )
            adj_metrics = evaluate_predictions(
                adj_df["label"], adj_df["y_pred"], adj_df["y_prob"]
            )
            rate_records.append(
                {
                    "rate_key": f"{rate:.2%}",
                    "pos_rate": float((adj_df["label"] == 1).mean()),
                    "total_rows": len(adj_df),
                    "pos_rows": int((adj_df["label"] == 1).sum()),
                    "neg_rows": int((adj_df["label"] == 0).sum()),
                    **adj_metrics,
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary: dict[str, dict[str, Any]] = {}
    for metric in CORE_METRICS:
        vals = per_seed_df[metric].to_numpy(dtype=float)
        if is_multi:
            mean, ci_low, ci_high = compute_confidence_interval(
                vals, confidence=confidence
            )
            half_width = 0.5 * (ci_high - ci_low)
            summary[metric] = {
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "half_width": half_width,
                "formatted": f"{mean:.4f} ± {half_width:.4f}",
            }
        else:
            val = float(vals[0])
            summary[metric] = {
                "mean": val,
                "ci_low": None,
                "ci_high": None,
                "half_width": None,
                "formatted": f"{val:.4f}",
            }

    rate_df = pd.DataFrame(rate_records)
    rate_order = ["original"] + [f"{r:.2%}" for r in target_rates]
    prefix = "Avg " if is_multi else ""
    rate_summary_rows: list[dict[str, Any]] = []
    for key in rate_order:
        sub = rate_df[rate_df["rate_key"] == key]
        label = (
            f"{sub['pos_rate'].mean():.2%} (Original)"
            if key == "original"
            else key
        )
        rate_summary_rows.append(
            {
                "Target Pos Rate": label,
                f"{prefix}Total": int(round(sub["total_rows"].mean())),
                f"{prefix}Pos (1)": int(round(sub["pos_rows"].mean())),
                f"{prefix}Neg (0)": int(round(sub["neg_rows"].mean())),
                **{
                    m: format_metric(sub[m], confidence=confidence)
                    for m in CORE_METRICS
                },
            }
        )
    rate_summary_df = pd.DataFrame(rate_summary_rows)

    print(f"\n================ {split_name} ================")
    if is_multi:
        print(
            f"Seeds (K={len(validated_seeds)}): {list(validated_seeds)} | "
            f"Avg Train: {int(round(per_seed_df['Train Rows'].mean())):,} | "
            f"Avg Test: {int(round(per_seed_df['Test Rows'].mean())):,}"
        )
        for metric in CORE_METRICS:
            s = summary[metric]
            print(
                f"  + {metric:<10}: {s['formatted']}  "
                f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
            )
        rate_header = f"{split_name}, Mean ± {confidence:.0%} CI"
    else:
        print(
            f"Seed: {validated_seeds[0]} | "
            f"Train: {int(per_seed_df['Train Rows'].iloc[0]):,} | "
            f"Test: {int(per_seed_df['Test Rows'].iloc[0]):,}"
        )
        for metric in CORE_METRICS:
            print(f"  + {metric:<10}: {summary[metric]['formatted']}")
        rate_header = split_name

    print(f"\n--- Realistic Positive-Rate Evaluation ({rate_header}) ---")
    print(rate_summary_df.to_string(index=False))

    return {
        "seeds": validated_seeds,
        "models": models,
        "per_seed_df": per_seed_df,
        "summary": summary,
        "rate_summary_df": rate_summary_df,
    }
