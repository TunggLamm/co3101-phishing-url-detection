"""Evaluation metrics and realistic positive-rate utilities for PhiUSIIL auditing.

The module provides:
- compute_pr_auc: step-wise Precision-Recall AUC (Average Precision) calculation;
- adjust_positive_rate: lowers positive class prevalence to a target rate while
  preserving 100% of the positive samples.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_pr_auc(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    *,
    pos_label: int = 1,
) -> float:
    """Compute Precision-Recall AUC (Average Precision) from first principles.

    Calculates the step-wise area under the Precision-Recall curve:
        AP = sum_k (R_k - R_{k-1}) * P_k
    where P_k and R_k are the precision and recall at the k-th unique score
    threshold sorted in descending order. Matches scikit-learn's
    ``average_precision_score``.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities or decision scores for the positive class.
        pos_label: Label value considered positive (default: 1 = Phishing).

    Returns:
        Precision-Recall AUC as a float in [0.0, 1.0].

    Raises:
        ValueError: If inputs are empty, have mismatched lengths, contain NaN
            values, or contain no positive samples.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    if y_true_arr.ndim != 1 or y_score_arr.ndim != 1:
        raise ValueError("y_true and y_score must be 1D sequences.")

    if len(y_true_arr) == 0:
        raise ValueError("y_true and y_score cannot be empty.")

    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} elements, "
            f"y_score has {len(y_score_arr)} elements."
        )

    if pd.isna(y_true_arr).any() or np.isnan(y_score_arr).any():
        raise ValueError("y_true and y_score must not contain NaN values.")

    binary_true = (y_true_arr == pos_label).astype(np.int64)
    total_positives = int(binary_true.sum())

    if total_positives == 0:
        raise ValueError(
            f"No positive samples found for pos_label={pos_label!r}; "
            "PR-AUC is undefined."
        )

    # Sort scores in descending order using a stable sort.
    desc_order = np.argsort(-y_score_arr, kind="mergesort")
    y_score_sorted = y_score_arr[desc_order]
    y_true_sorted = binary_true[desc_order]

    # Evaluate precision and recall only at the last index of each unique threshold
    # so tied scores are grouped together exactly as in scikit-learn.
    distinct_value_indices = np.where(np.diff(y_score_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_score_sorted) - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    predicted_positives = 1 + threshold_idxs

    precision = tps / predicted_positives
    recall = tps / total_positives

    recall_steps = np.diff(np.r_[0.0, recall])
    pr_auc = float(np.sum(recall_steps * precision))

    return pr_auc


def adjust_positive_rate(
    df: pd.DataFrame,
    target_rate: float,
    *,
    label_col: str = "label",
    pos_label: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Lower the positive class rate to ``target_rate`` while keeping all positives.

    To simulate realistic phishing prevalence (e.g., 1% or 5%) without losing
    any phishing attack examples, 100% of positive rows are kept intact and
    negative rows are sampled to achieve the target positive fraction:
        N_neg_required = round(N_pos * (1 - target_rate) / target_rate)

    Args:
        df: Input DataFrame containing positive and negative rows.
        target_rate: Desired positive class fraction in (0, current_rate).
        label_col: Column name containing class labels.
        pos_label: Value in ``label_col`` representing the positive class.
        random_state: Random seed for reproducible negative sampling.

    Returns:
        New DataFrame with positive prevalence adjusted to ``target_rate`` and
        all original positive samples preserved.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot adjust positive rate of an empty DataFrame.")
    if label_col not in df.columns:
        raise ValueError(f"Missing required label column: {label_col!r}")
    if df[label_col].isna().any():
        raise ValueError(f"Label column {label_col!r} contains missing values.")
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must be between 0 and 1 (exclusive).")

    pos_mask = df[label_col] == pos_label
    pos_df = df.loc[pos_mask].copy()
    neg_df = df.loc[~pos_mask].copy()

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    if n_pos == 0:
        raise ValueError(
            f"No positive samples found for pos_label={pos_label!r}."
        )
    if n_neg == 0:
        raise ValueError(
            "No negative samples found to adjust the positive rate."
        )

    current_rate = n_pos / len(df)
    if target_rate >= current_rate:
        raise ValueError(
            f"target_rate ({target_rate:.4f}) must be strictly less than "
            f"current positive rate ({current_rate:.4f})."
        )

    n_neg_required = int(round(n_pos * (1.0 - target_rate) / target_rate))

    sampled_neg_df = neg_df.sample(
        n=n_neg_required,
        replace=True,
        random_state=random_state,
    )

    adjusted_df = pd.concat([pos_df, sampled_neg_df], axis=0, ignore_index=True)
    return adjusted_df
