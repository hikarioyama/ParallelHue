import pytest

from parallelhue.summary import stream_rate_stats


def test_rates_and_mean_tokens_default_case():
    aggregate, mean_rate, mean_tokens = stream_rate_stats(8192, 28.68, 16)
    assert aggregate == pytest.approx(8192 / 28.68)
    assert mean_rate == pytest.approx((8192 / 28.68) / 16)
    assert mean_tokens == pytest.approx(512.0)


def test_no_generation_means_all_unavailable():
    aggregate, mean_rate, mean_tokens = stream_rate_stats(None, 28.68, 16)
    assert aggregate is None
    assert mean_rate is None
    assert mean_tokens is None


def test_concurrency_one_makes_mean_rate_equal_aggregate():
    aggregate, mean_rate, _ = stream_rate_stats(8192, 28.68, 1)
    assert mean_rate == pytest.approx(aggregate)


def test_zero_makespan_keeps_mean_tokens_but_drops_rates():
    aggregate, mean_rate, mean_tokens = stream_rate_stats(8192, 0.0, 16)
    assert aggregate is None
    assert mean_rate is None
    assert mean_tokens == pytest.approx(512.0)
