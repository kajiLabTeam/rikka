import pytest

from rikka.config import WEINBERG_REFERENCE_K, compute_weinberg_k
from rikka.ping import ping


def test_ping_returns_expected_message() -> None:
    assert ping() == "Hello, rikka"


def test_compute_weinberg_k_scales_with_height() -> None:
    assert compute_weinberg_k(1.70) == pytest.approx(WEINBERG_REFERENCE_K)
    assert compute_weinberg_k(1.80) > compute_weinberg_k(1.60)


def test_compute_weinberg_k_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError, match="height_m"):
        compute_weinberg_k(0.0)
