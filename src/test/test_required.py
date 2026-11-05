"""Consolidated suite for the 3 mandatory unit tests required by Section 6."""

import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent.parent

for path_str in (str(PROJECT_DIR), str(TEST_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from test_data_split import test_etld1_split_has_no_domain_overlap
from test_utils import (
    test_adjust_positive_rate_matches_target_and_preserves_all_positives,
    test_compute_pr_auc_matches_sklearn_on_known_vector,
)

__all__ = [
    "test_etld1_split_has_no_domain_overlap",
    "test_compute_pr_auc_matches_sklearn_on_known_vector",
    "test_adjust_positive_rate_matches_target_and_preserves_all_positives",
]
