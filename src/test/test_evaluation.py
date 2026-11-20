"""Tests for single-seed and multi-seed evaluation with optional 95% CIs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_split import etld1_split, random_split
from src.evaluation import (
    CORE_METRICS,
    DEFAULT_SEEDS,
    evaluate_model,
    normalize_seeds,
)
from src.train_gb import build_gb_model, print_feature_importances
from src.train_lr import build_logistic_regression_pipeline, print_feature_weights
from src.utils import compute_confidence_interval


def make_synthetic_dataset(seed: int = 123) -> pd.DataFrame:
    """Build a small multi-domain dataset with the 10 lexical features."""
    rng = np.random.default_rng(seed)
    suffixes = ("com", "net", "org", "co.uk")
    rows = []

    for d_idx in range(30):
        domain_is_phish = d_idx % 2 == 0
        domain = f"site-{d_idx}.{suffixes[d_idx % len(suffixes)]}"
        for u_idx in range(8):
            is_phish = domain_is_phish if rng.random() > 0.20 else (not domain_is_phish)
            rows.append(
                {
                    "URL": f"https://sub{u_idx}.{domain}/path/{u_idx}",
                    "label": 0 if is_phish else 1,  # UCI raw convention
                    "URLLength": max(10.0, float(rng.normal(52 if is_phish else 36, 18))),
                    "DomainLength": max(5.0, float(rng.normal(20 if is_phish else 15, 7))),
                    "IsDomainIP": int(is_phish and rng.random() < 0.25),
                    "NoOfSubDomain": int(rng.integers(1, 4) if is_phish else rng.integers(0, 3)),
                    "HasObfuscation": int(is_phish and rng.random() < 0.30),
                    "LetterRatioInURL": float(np.clip(rng.normal(0.60 if is_phish else 0.68, 0.12), 0.1, 0.95)),
                    "DegitRatioInURL": float(np.clip(rng.normal(0.14 if is_phish else 0.08, 0.08), 0.0, 0.6)),
                    "NoOfEqualsInURL": int(rng.integers(0, 4) if is_phish else rng.integers(0, 2)),
                    "NoOfQMarkInURL": int(rng.integers(0, 3) if is_phish else rng.integers(0, 2)),
                    "SpacialCharRatioInURL": float(np.clip(rng.normal(0.19 if is_phish else 0.14, 0.08), 0.01, 0.5)),
                }
            )

    return pd.DataFrame(rows)


def test_normalize_seeds_accepts_single_or_multiple_distinct_ints() -> None:
    assert normalize_seeds(42) == (42,)
    assert normalize_seeds([42]) == (42,)
    assert normalize_seeds([42, 43]) == (42, 43)
    assert normalize_seeds(DEFAULT_SEEDS) == DEFAULT_SEEDS

    with pytest.raises(ValueError, match="At least 1 seed"):
        normalize_seeds([])

    with pytest.raises(ValueError, match="distinct"):
        normalize_seeds([42, 42])

    with pytest.raises(ValueError, match="integer"):
        normalize_seeds(True)

    with pytest.raises(ValueError, match="integers"):
        normalize_seeds([True, 2])


def test_evaluate_model_single_seed_runs_without_ci() -> None:
    df = make_synthetic_dataset()

    res_int = evaluate_model(
        df,
        split_name="SINGLE SEED INT",
        split_fn=lambda d, s: random_split(d, test_size=0.25, random_state=s),
        model_builder=build_logistic_regression_pipeline,
        seeds=42,
        target_rates=(0.10,),
    )
    res_seq = evaluate_model(
        df,
        split_name="SINGLE SEED SEQ",
        split_fn=lambda d, s: random_split(d, test_size=0.25, random_state=s),
        model_builder=build_logistic_regression_pipeline,
        seeds=[42],
        target_rates=(0.10,),
    )

    assert res_int["seeds"] == (42,)
    assert len(res_int["models"]) == 1
    pd.testing.assert_frame_equal(res_int["per_seed_df"], res_seq["per_seed_df"])

    for metric in CORE_METRICS:
        s = res_int["summary"][metric]
        assert s["ci_low"] is None
        assert s["ci_high"] is None
        assert s["half_width"] is None
        assert "±" not in s["formatted"]
        assert s["formatted"] == f"{s['mean']:.4f}"

    # Verify feature reporting works cleanly for single-seed models
    print_feature_weights(res_int["models"])


def test_evaluate_model_multi_seed_matches_compute_confidence_interval() -> None:
    df = make_synthetic_dataset()
    seeds = (10, 20, 30, 40, 50)

    res = evaluate_model(
        df,
        split_name="TEST LR eTLD+1",
        split_fn=lambda d, s: etld1_split(d, test_size=0.25, random_state=s, n_candidates=20),
        model_builder=build_logistic_regression_pipeline,
        seeds=seeds,
        target_rates=(0.10, 0.05),
        assert_etld1_disjoint=True,
    )

    assert len(res["models"]) == 5
    assert res["per_seed_df"]["Seed"].tolist() == list(seeds)
    assert len(res["rate_summary_df"]) == 3  # original + 10% + 5%

    for metric in CORE_METRICS:
        expected_mean, expected_low, expected_high = compute_confidence_interval(
            res["per_seed_df"][metric], confidence=0.95
        )
        expected_half = 0.5 * (expected_high - expected_low)
        actual = res["summary"][metric]

        assert actual["mean"] == pytest.approx(expected_mean, abs=1e-12)
        assert actual["ci_low"] == pytest.approx(expected_low, abs=1e-12)
        assert actual["ci_high"] == pytest.approx(expected_high, abs=1e-12)
        assert actual["formatted"] == f"{expected_mean:.4f} ± {expected_half:.4f}"

    assert res["summary"]["F1-score"]["half_width"] > 0.0


def test_evaluate_model_gb_single_and_multi_seed_and_overlap_check() -> None:
    df = make_synthetic_dataset()
    split_fn = lambda d, s: random_split(d, test_size=0.25, random_state=s)

    gb_single = evaluate_model(
        df, split_name="GB SINGLE", split_fn=split_fn, model_builder=build_gb_model,
        seeds=42, target_rates=(0.10,),
    )
    assert len(gb_single["models"]) == 1
    print_feature_importances(gb_single["models"])

    gb_two = evaluate_model(
        df, split_name="GB TWO", split_fn=split_fn, model_builder=build_gb_model,
        seeds=(42, 43), target_rates=(0.10,),
    )
    assert len(gb_two["models"]) == 2
    assert "±" in gb_two["summary"]["F1-score"]["formatted"]

    with pytest.raises(AssertionError, match=r"eTLD\+1 overlap detected"):
        evaluate_model(
            df,
            split_name="LEAKY",
            split_fn=split_fn,
            model_builder=build_logistic_regression_pipeline,
            seeds=42,
            target_rates=(0.10,),
            assert_etld1_disjoint=True,
        )
