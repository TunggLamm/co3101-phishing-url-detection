"""Tests for random and eTLD+1 train/test splitting."""

import pandas as pd
import pytest

from src.data_split import (
    assert_no_etld1_overlap,
    etld1_split,
    extract_etld1,
    random_split,
)


def make_toy_dataset() -> pd.DataFrame:
    """Build exactly 20 rows across five registrable domains."""
    domains = ["alpha.com", "beta.net", "gamma.org", "delta.co.uk", "epsilon.com"]
    subdomains = ["login", "shop", "account", "help"]
    labels_per_domain = [0, 0, 1, 1]  # original PhiUSIIL labels for test purposes

    rows = []
    for domain in domains:
        for subdomain, label in zip(subdomains, labels_per_domain):
            rows.append(
                {
                    "URL": f"https://{subdomain}.{domain}/page",
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def test_extract_etld1_handles_common_suffixes() -> None:
    assert extract_etld1("https://login.example.com/path") == "example.com"
    assert extract_etld1("https://a.b.example.co.uk/path") == "example.co.uk"


def test_random_split_keeps_all_rows_and_is_reproducible() -> None:
    df = make_toy_dataset()

    train_1, test_1 = random_split(df, test_size=0.25, random_state=7)
    train_2, test_2 = random_split(df, test_size=0.25, random_state=7)

    assert len(train_1) + len(test_1) == len(df)
    assert set(train_1.index).isdisjoint(set(test_1.index))
    assert train_1.index.tolist() == train_2.index.tolist()
    assert test_1.index.tolist() == test_2.index.tolist()


def test_etld1_split_has_no_domain_overlap_and_keeps_all_rows() -> None:
    df = make_toy_dataset()

    train_df, test_df = etld1_split(
        df,
        test_size=0.4,
        random_state=42,
        url_col="URL",
    )

    train_domains = {extract_etld1(url) for url in train_df["URL"]}
    test_domains = {extract_etld1(url) for url in test_df["URL"]}

    assert train_domains.isdisjoint(test_domains)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.index).isdisjoint(set(test_df.index))
    assert_no_etld1_overlap(train_df, test_df, url_col="URL")


def test_overlap_validator_fails_when_domain_is_in_both_splits() -> None:
    train_df = pd.DataFrame({"URL": ["https://login.example.com/page"]})
    test_df = pd.DataFrame({"URL": ["https://shop.example.com/page"]})

    with pytest.raises(AssertionError, match=r"eTLD\+1 overlap detected"):
        assert_no_etld1_overlap(train_df, test_df, url_col="URL")


def test_etld1_split_rejects_missing_url() -> None:
    df = make_toy_dataset()
    df.loc[0, "URL"] = None

    with pytest.raises(ValueError, match="missing URL"):
        etld1_split(df, test_size=0.4, random_state=42)
