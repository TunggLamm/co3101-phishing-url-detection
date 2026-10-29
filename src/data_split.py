"""Train/test splitting utilities for the PhiUSIIL phishing URL project.

The module provides:
- random_split: ordinary row-level random splitting, stratified by label by default;
- etld1_split: splitting that keeps each eTLD+1 group entirely in train or test;
- extract_etld1 and assert_no_etld1_overlap: helpers for extraction and validation.

The original dataset labels are not changed by these functions.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit, train_test_split


INCLUDE_PSL_PRIVATE_DOMAINS = False

# Use tldextract's bundled Public Suffix List snapshot to avoid an HTTP fetch during the first run. 
# set INCLUDE_PSL_PRIVATE_DOMAINS to include private suffixes so hosted subdomains can be separated as their own registrable-domain groups.
_TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=INCLUDE_PSL_PRIVATE_DOMAINS,
)


def extract_etld1(url: object) -> str:
    """Return a stable grouping key for a URL's eTLD+1.

    For normal domain names, returns values such as ``example.com`` or
    ``example.co.uk``. IP addresses are grouped by the IP itself (prefixed
    with ``ip:``), since IP addresses do not have an eTLD+1. Single-label
    hostnames such as ``localhost`` are grouped by that hostname.

    Raises:
        ValueError: if the URL is missing, blank, has no hostname, or has a
            dotted hostname for which the public suffix cannot be resolved.
    """
    if pd.isna(url):
        raise ValueError("Cannot extract eTLD+1 from a missing URL.")

    raw_url = str(url).strip()
    if not raw_url:
        raise ValueError("Cannot extract eTLD+1 from a blank URL.")

    candidate = raw_url if "://" in raw_url or raw_url.startswith("//") else f"//{raw_url}"

    try:
        hostname = urlsplit(candidate).hostname
    except ValueError as exc:
        raise ValueError(f"Could not parse hostname from URL: {raw_url!r}") from exc

    if not hostname:
        raise ValueError(f"Could not find a hostname in URL: {raw_url!r}")

    hostname = hostname.rstrip(".").lower()

   # IP addresses handling
    try:
        ip_address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return f"ip:{ip_address.compressed}"

    extracted = _TLD_EXTRACTOR(hostname)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()

    # The hostname itself may be a private suffix
    # It has no additional registrable label, so group this exact hostname.
    if extracted.suffix and hostname == extracted.suffix:
        return f"host:{hostname}"

    # A single-label host has no eTLD+1, use its full hostname as a group.
    if "." not in hostname:
        return f"host:{hostname}"

    # Unidentifiable dotted hostnames
    raise ValueError(
        f"Could not determine eTLD+1 for hostname {hostname!r}. "
        "Check the URL and Public Suffix List assumptions before splitting."
    )


def _check_dataframe(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Validate common input requirements before splitting."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def random_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows randomly into train and test."""
    _check_dataframe(df, {label_col})

    if df[label_col].isna().any():
        raise ValueError(
            f"Label column {label_col!r} contains missing values."
        )

    if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1 (exclusive).")

    stratify_values = df[label_col] if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    return train_df.copy(), test_df.copy()


def assert_no_etld1_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    url_col: str = "URL",
) -> None:
    """Raise AssertionError if an eTLD+1 occurs in both splits."""
    _check_dataframe(train_df, {url_col})
    _check_dataframe(test_df, {url_col})

    train_domains = {
        extract_etld1(url) for url in train_df[url_col]
    }
    test_domains = {
        extract_etld1(url) for url in test_df[url_col]
    }

    overlap = train_domains.intersection(test_domains)

    if overlap:
        examples = sorted(overlap)[:10]
        raise AssertionError(
            "eTLD+1 overlap detected between train and test. "
            f"Example overlapping groups: {examples}"
        )


def etld1_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    url_col: str = "URL",
    n_candidates: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by eTLD+1, keeping each domain group in one split."""
    _check_dataframe(df, {url_col})

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1.")

    groups = pd.Series(
        [extract_etld1(url) for url in df[url_col]],
        index=df.index,
        name="_etld1_group",
    )

    if groups.nunique() < 2:
        raise ValueError(
            "At least two distinct eTLD+1 groups are required for a split."
        )

    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )

    best_indices = None
    best_difference = float("inf")

    # Try several group splits and choose the closest test-row proportion.
    for train_positions, test_positions in splitter.split(
        df, groups=groups.to_numpy()
    ):
        actual_fraction = len(test_positions) / len(df)
        difference = abs(actual_fraction - test_size)

        if difference < best_difference:
            best_difference = difference
            best_indices = (train_positions, test_positions)

    if best_indices is None:
        raise RuntimeError("Could not generate an eTLD+1 train/test split.")

    train_positions, test_positions = best_indices

    train_df = df.iloc[train_positions].copy()
    test_df = df.iloc[test_positions].copy()

    assert_no_etld1_overlap(train_df, test_df, url_col=url_col)

    return train_df, test_df
