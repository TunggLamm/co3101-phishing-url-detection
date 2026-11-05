"""Tests for PR-AUC calculation and positive-rate adjustment (Mandatory Tests #2 and #3)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.utils import adjust_positive_rate, compute_pr_auc


def test_compute_pr_auc_matches_sklearn_on_known_vector() -> None:
    """Mandatory Test #2: PR-AUC function matches sklearn on a known vector."""
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.10, 0.40, 0.35, 0.80, 0.65, 0.20, 0.90, 0.50]

    expected = average_precision_score(y_true, y_score)
    actual = compute_pr_auc(y_true, y_score)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_compute_pr_auc_matches_sklearn_with_tied_scores() -> None:
    """Verify PR-AUC matches sklearn when multiple samples share identical scores."""
    y_true = [1, 0, 1, 0, 1, 0, 0, 1]
    y_score = [0.9, 0.5, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]

    expected = average_precision_score(y_true, y_score)
    actual = compute_pr_auc(y_true, y_score)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_compute_pr_auc_rejects_invalid_inputs() -> None:
    """Verify input validation for empty, mismatched, or all-negative inputs."""
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_pr_auc([], [])

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_pr_auc([0, 1], [0.2])

    with pytest.raises(ValueError, match="No positive samples"):
        compute_pr_auc([0, 0, 0], [0.1, 0.4, 0.9])


def test_adjust_positive_rate_matches_target_and_preserves_all_positives() -> None:
    """Mandatory Test #3: Lower positive rate while keeping 100% of positive samples."""
    n_pos = 40
    n_neg = 60
    pos_urls = [f"https://phish-{i}.bad-example.com/login" for i in range(n_pos)]
    neg_urls = [f"https://www.legit-{i}.example.org/home" for i in range(n_neg)]

    df_synthetic = pd.DataFrame(
        {
            "URL": pos_urls + neg_urls,
            "label": [1] * n_pos + [0] * n_neg,
        }
    )

    target_rate = 0.05  # Lower positive rate from 40% down to 5%
    adjusted_df = adjust_positive_rate(
        df_synthetic,
        target_rate=target_rate,
        label_col="label",
        pos_label=1,
        random_state=42,
    )

    adjusted_positives = adjusted_df[adjusted_df["label"] == 1]
    actual_rate = len(adjusted_positives) / len(adjusted_df)

    # (a) Resulting positive rate matches the target rate.
    assert actual_rate == pytest.approx(target_rate, abs=1e-4)

    # (b) Invariant: 100% of original positive samples are preserved intact.
    assert len(adjusted_positives) == n_pos
    assert set(adjusted_positives["URL"]) == set(pos_urls)
    assert adjusted_positives["URL"].tolist() == pos_urls


def test_adjust_positive_rate_is_reproducible_and_validates_target_rate() -> None:
    """Verify deterministic sampling and rejection of invalid target rates."""
    df = pd.DataFrame(
        {
            "URL": [f"https://site-{i}.com" for i in range(20)],
            "label": [1] * 8 + [0] * 12,  # 40% positive rate
        }
    )

    adj_1 = adjust_positive_rate(df, target_rate=0.10, random_state=7)
    adj_2 = adjust_positive_rate(df, target_rate=0.10, random_state=7)
    assert adj_1["URL"].tolist() == adj_2["URL"].tolist()

    with pytest.raises(ValueError, match="strictly less than current positive rate"):
        adjust_positive_rate(df, target_rate=0.50)

    with pytest.raises(ValueError, match="between 0 and 1"):
        adjust_positive_rate(df, target_rate=0.0)
